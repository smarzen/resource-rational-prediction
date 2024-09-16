import numpy as np
import pylab as pl
import scipy as sp
import scipy.optimize as spo
import csv
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras

file_path = 'C:/Users/amyyu/Downloads/Research Summer 2023/Experiment3a.csv' 
emlibrary_path = 'EMLibrary/'

## trains an LSTM off the given parameters.
## the LSTM will train individually on each series of 15 timesteps
## we take the log probability score of each of the participant's guesses against the model's guess
def logL_LSTM(data_input,data_guesses,num_nodes=10,lr=1.2e-2,num_classes=2,timesteps=15,num_epochs=30):
	untrained_indexes=[]
	# make an LSTM and train it
	logL = 0
	for t in range(timesteps+1,len(data_input)):
		tf.keras.backend.clear_session()
		batch_x = []; batch_y = []
		# preprocessing prediction input
		for i in range(timesteps,t):
			# use the last timesteps data to predct the next time series data
			batch_x.append([int(data_input[j]) for j in range(i-timesteps,i)])
			batch_y.append([int(data_input[i])])
		batch_x = np.asarray(batch_x)
		batch_y = np.asarray(batch_y)
		batch_x = batch_x.reshape((len(batch_x),timesteps,1))
		batch_x = tf.convert_to_tensor(batch_x)
		# use keras to make the LSTM
		X = tf.keras.Input(name='X', shape=[timesteps, 1], dtype=tf.dtypes.float32)
		# model training
		lstm_output = tf.keras.layers.LSTM(num_nodes)(X)
		prediction = tf.keras.layers.Dense(num_classes,activation='softmax')(lstm_output)
		# use Adam to train the LSTM
		opt = tf.keras.optimizers.Adam(learning_rate=lr)
		model = keras.Model(inputs=X, outputs=prediction)
		# use cross entropy to train the model
		model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
		hist = model.fit(x=batch_x, y=batch_y, epochs=num_epochs, verbose = 0)
		# checking to see if the model has trained properly by evaluating if the loss has gone up 
		# this was useful for fiddling with the learning rate
		if hist.history['loss'][-1] - hist.history['loss'][0] >= 0:
			untrained_indexes.append(t)
		# look at the model's predictions for the last symbol in the sequence
		xpreds = model.predict(batch_x[-1])
		# add log probability of a guess depending on model prediction to ongoing sum
		# we calculate the log probability based on what the participant guessed compared to the model
		# we use probability matching to make a log likelihood that isn't just, did you get it right
		if data_guesses[t]=='0':
			logL += np.log(np.exp(xpreds[-1][0])/(np.exp(xpreds[-1][0])+np.exp(xpreds[-1][1])))
		else:
			logL += np.log(np.exp(xpreds[-1][1])/(np.exp(xpreds[-1][0])+np.exp(xpreds[-1][1])))
	return untrained_indexes, logL

def logL_ngram_avg(alpha,beta,gamma,data_input,data_guesses):
	# run through all machines between 1 and 3 states
	Rs = np.arange(1,6)
	# probability over models
	P0Rs = np.exp(-gamma*2**Rs) # gamma=1
	P0Rs /= np.sum(P0Rs)
	# log likelihood of seeing the behavior we do under this cognition model over all timesteps
	logL_ts = 0
	# log likelihood over all models
	logL = 0
	# looping over timesteps 
	# use the formula in the Entropy paper, https://www.mdpi.com/1099-4300/22/8/896
	for t in range(len(data_input)-np.max(Rs)):
		Foo = np.zeros(len(Rs))
		for R in Rs:
			# find the number of instances in the data so far
			# n_foo2 represents the number of instances of a correct guess
			n_foo = 0; n_foo2 = 0
			for i in range(t):
				if data_input[i:i+R]==data_input[t:t+R]:
					n_foo += 1
					if data_guesses[t+R]==data_input[i+R]:
						n_foo2 += 1
			Foo[R-1] = (alpha+beta*n_foo2)/(2*alpha+beta*n_foo)
		logL = np.inner(P0Rs,Foo)
		logL_ts += np.log(logL)
	return logL_ts

def logL_BSI_argmax(alpha,beta,gamma,data_input,data_guesses,hidden_states):
	# run through all eMs
	P0s = []
	transition_fnctns = {}
	for i in range(3):
		count = 1
		f = open(emlibrary_path+str(i+1)+'.hs')
		eM = f.readline()
		while eM!='':
			# eM is the line of the eM name in the file
			# convert the eM topology to a transition function between states
			eM0 = eM
			eM = eM[3:-1]
			x = eM.find(']')
			Ss = np.arange(int(eM[x-1]))
			Xs = np.arange(2)
			eM = eM[x+3:-x-3]
			transition_fnctn = {}
			while len(eM)>=7:
				foo = eM[:7]
				transition_fnctn[str(int(foo[1])-1)+foo[3]] = int(foo[5])-1
				# get rid of the one you just did from the eM description
				if len(eM)>8:
					eM = eM[8:]
				else:
					eM = []
			transition_fnctns[eM0] = transition_fnctn
			# find the model probability
			P0s.append(np.exp(-gamma*len(transition_fnctn)))
			eM = f.readline()
	# normalize the model probability
	P0s = np.asarray(P0s); P0s /= np.sum(P0s)
	# log likelihood of seeing the behavior we do under this cognition model over all timesteps
	# using the formula in the SI of the paper to which this is attached
	logL_ts = 0
	for t in range(1,len(data_input)):
		Foo = []
		for eM in transition_fnctns:
			# find the number of instances in the data so far
			n_foo = 0; n_foo2 = 0
			for i in range(1,t):
				if hidden_states[i-1]==hidden_states[t-1]: # fix this
					n_foo += 1
					if data_guesses[t]==data_input[i]:
						n_foo2 += 1
			Foo.append((alpha+beta*n_foo2)/(2*alpha+beta*n_foo))
		logL = np.inner(P0s,np.asarray(Foo))
		logL_ts += np.log(logL)
	return logL_ts

def logL_GLM(k,data_input,data_guesses):
	# pad everything with two false data points that will get overwhelmed.
	# run through all data points
	# evaluate the log likelihood of logistic regression being used by the participant
	# train a logistic regression on the data and use probability matching to determine the log likelihood
	for t in range(len(data_input)-k):
		# get the data listed
		x_train = np.asarray([np.asarray([int(data_input[j]) for j in range(t2,t2+k)]) for t2 in range(t+1)])
		y_train = np.asarray([int(data_input[t2]) for t2 in range(k,t+1+k)])
		# pad
		x_train = np.vstack([x_train,np.zeros([2,k])])
		y_train = np.hstack([y_train,np.asarray([0,1])])
		# train logistic regression on the data
		reg2 = LogisticRegression(penalty='l2').fit(x_train,y_train)
		# get the probability of the next symbol
		pred_log_prob = reg2.predict_log_proba(x_train)
		ys_true = np.asarray([int(data_guesses[j]) for j in range(k,t+1+k)])
		# calculate the log probability
		pred_log_prob = np.asarray([pred_log_prob[i,ys_true[i]] for i in range(len(ys_true))])
	return np.sum(pred_log_prob)

Inputs = []
Guesses = []
ParticipantIDs = []
Input_Types = []
hidden_states = []

with open(file_path) as csvfile:
		spamreader = csv.reader(csvfile)
		# add values from csv file to list
		for row in spamreader:
			input_ = row[7]
			guess_ = row[6]
			ID = row[0]
			hmm_states = row[13]
			Inputs.append(input_)
			Guesses.append(guess_)
			ParticipantIDs.append(ID)
			Input_Types.append(row[15])
			hidden_states.append(hmm_states)
		#remove field names
		Inputs.pop(0)
		Guesses.pop(0)
		ParticipantIDs.pop(0)
		Input_Types.pop(0)
		hidden_states.pop(0)

# this function writes the log probability scores of each model for each participant into an output .txt file
def analyzeData(inputs, guesses, participants, input_types):
	for i in range(len(inputs)):
		f = open(input_types[i]+"Data.txt","a+")
		# calls function to calculate the log probability score for LSTM
		untrained, lstm = logL_LSTM(inputs[i], guesses[i])
		# calls function to calculate n-gram model log probability score
		ngram_avg = spo.minimize(lambda x: -logL_ngram_avg(1,x[0],x[1],inputs[i],guesses[i]),[.1,.1],bounds=((0,1),(0,100)))
		ngram_avg = -ngram_avg.fun
		# function call for BSI score calculation
		bsi = spo.minimize(lambda x: -logL_BSI_argmax(1,x[0],x[1],inputs[i],guesses[i],hidden_states[i]),[.1,.1],bounds=((0,1),(0,100)))
		bsi = -bsi.fun
		glms = []
		# calculate the log probability score of the GLM
		for k in range(1,11):
			glms.append(logL_GLM(k,inputs[i],guesses[i]))
			glm = min(glms)
		# writing the scores of the participant into the file
		# we flag whether or not the LSTM trained adequately within this file based off of the loss score
		if len(untrained) > 0:
			newline = f'\n\n{participants[i]} ({input_types[i]}) \nngram-avg: {ngram_avg}\nBSI: {bsi}\nGLM: {glm}\nLSTM: {lstm} (Untrained at: {untrained})'
		elif len(untrained) == 0:
			newline = f'\n\n{participants[i]} ({input_types[i]}) \nngram-avg: {ngram_avg}\nBSI: {bsi}\nGLM: {glm}\nLSTM: {lstm} (Trained)'
		f.write(newline)
		print("completed row ", i)
		f.close()

analyzeData(Inputs, Guesses, ParticipantIDs, Input_Types)