import numpy as np
import pylab as pl
import scipy as sp
import json
import csv
import seaborn as sns

# Should have accuracy, input, and output
# Type of process as well

# Will use the prediction as the R -- could instead use the last m symbols as R
# Rationale is that the prediction contains implicit memory
# Downside is that we'll never see this memory
# Will use PRA rather than PIB

def emissionProbs(input_type='NoisyPeriodic'):
	if input_type=='NoisyPeriodic':
		q = 0.1
		p = np.zeros([2,2])
		p[0,0] = 1
		p[1,0] = q
		p[1,1] = 1-q
		p = p*0.5
	elif input_type=='EvenProcess':
		q = 0.3
		p = np.zeros([2,2])
		p[0,0] = (1-q)/(1+q)
		p[0,1] = q/(1+q)
		p[1,1] = 1-(1/(1+q))
	else:
		q = 0.3
		p = np.zeros([3,2])
		p[0,0] = 1-q
		p[0,1] = q
		p[1,0] = 1
		p[2,1] = 1
		cl_psigma = np.array([10/3,1,1])
		cl_psigma = cl_psigma/(16/3)
		p[0,:] = p[0,:]*cl_psigma[0]
		p[1,:] = p[1,:]*cl_psigma[1]
		p[2,:] = p[2,:]*cl_psigma[2]
	return p

def rate_distortion(p, beta, tol=1e-6, max_iter=10000):
    """
    Calculate the rate-distortion function for a given joint probability distribution p and a beta value.

    Args:
    p: Joint probability distribution.
    beta: Beta value, which can be interpreted as the inverse temperature in statistical physics.
    tol: Tolerance for convergence. If the change in marginal_pXhat and conditional_pXhat_given_S is less than tol,
         the function will stop iterating and return the current rate and distortion.
    max_iter: Maximum number of iterations.

    Returns:
    R: The rate value in the rate-distortion function.
    D: The distortion value in the rate-distortion function.
    """
    
    # Calculate the marginal probabilities of X and S
    marginal_pX = np.sum(p, 0)
    marginal_pS = np.sum(p, 1)

    # Calculate conditional probability of X given S
    conditional_pX_given_S = np.dot(np.diag(1 / marginal_pS), p)

    # Distortion matrix is equal to the conditional probability of X given S
    distortion_matrix = conditional_pX_given_S

    # Initialize the conditional probability of Xhat given S randomly
    initial_conditional_pXhat_given_S0 = np.ones(len(marginal_pS))/len(marginal_pS)
    conditional_pXhat_given_S = np.vstack([initial_conditional_pXhat_given_S0, 1 - initial_conditional_pXhat_given_S0]).T

    # Calculate the marginal probability of Xhat
    marginal_pXhat = np.dot(conditional_pXhat_given_S.T, marginal_pS)

    # Iterate to refine the conditional probability of Xhat given S and marginal probability of Xhat
    for _ in range(max_iter):
        # Calculate new estimates
        log_conditional_pXhat_given_S = np.meshgrid(np.log2(marginal_pXhat), np.ones(len(marginal_pS)))[0] + beta * distortion_matrix
        new_conditional_pXhat_given_S = np.exp(log_conditional_pXhat_given_S)
        normalization_constants = np.sum(new_conditional_pXhat_given_S, 1)
        new_conditional_pXhat_given_S = np.dot(np.diag(1 / normalization_constants), new_conditional_pXhat_given_S)
        new_marginal_pXhat = np.dot(new_conditional_pXhat_given_S.T, marginal_pS)
        
        # Check for convergence
        if np.allclose(marginal_pXhat, new_marginal_pXhat, atol=tol) and np.allclose(conditional_pXhat_given_S, new_conditional_pXhat_given_S, atol=tol):
            # print('beta=', beta, 'converged after', _+1, 'iterations')
            break
        
        # Update estimates
        marginal_pXhat = new_marginal_pXhat
        conditional_pXhat_given_S = new_conditional_pXhat_given_S

    # Calculate the rate value R in the rate-distortion function
    R = -np.nansum(marginal_pXhat * np.log2(marginal_pXhat)) + np.dot(marginal_pS, np.nansum(conditional_pXhat_given_S * np.log2(conditional_pXhat_given_S), 1))

    # Calculate the distortion value D in the rate-distortion function
    D = np.dot(marginal_pS, np.sum(conditional_pXhat_given_S * conditional_pX_given_S, 1))

    return R, D

def rate(hidden_states,guesses):
	# get statistics
	# this only works when things are well-sampled
	joint_xy = {}
	marginal_x = {}
	marginal_y = {}
	for i in range(len(hidden_states)):
		xy = str(hidden_states[i])+str(guesses[i])
		x = str(hidden_states[i])
		y = str(guesses[i])
		if xy in joint_xy:
			joint_xy[xy] = joint_xy[xy]+1
		else:
			joint_xy[xy] = 1
		if x in marginal_x:
			marginal_x[x] = marginal_x[x]+1
		else:
			marginal_x[x] = 1
		if y in marginal_y:
			marginal_y[y] = marginal_y[y]+1
		else:
			marginal_y[y] = 1
	# hack to add appropriate pseudocounts
	all_cs = np.unique(hidden_states)
	all_output = [0,1]
	if len(marginal_y)<2:
		marginal_y[str(2)] = 0
	for cs in all_cs:
		for output in all_output:
			xy = str(cs)+str(output)
			if xy in joint_xy:
				pass
			else:
				joint_xy[xy] = 0
	# find tot
	tot_xy = np.sum(list(joint_xy.values()))
	tot_x = np.sum(list(marginal_x.values()))
	tot_y = np.sum(list(marginal_y.values()))
	# get the entropies
	p_xy = np.asarray(list(joint_xy.values()))/tot_xy
	H_xy = -np.nansum(p_xy*np.log2(p_xy))
	p_x = np.asarray(list(marginal_x.values()))/tot_x
	H_x = -np.nansum(p_x*np.log2(p_x))
	p_y = np.asarray(list(marginal_y.values()))/tot_y
	H_y = -np.nansum(p_y*np.log2(p_y))
	# get the mutual information
	I_xy = H_x+H_y-H_xy
	return I_xy

# Input your own file path to https://github.com/vanferdi/resource-rational-prediction/blob/main/Data/experiment.csv
file_path = 'experiment.csv'

Inputs = []
Guesses = []
ParticipantIDs = []
Input_Types = []
Hidden_states = []
Accuracy = []
Bonus_round = []

with open(file_path) as csvfile:
		spamreader = csv.reader(csvfile)
		for row in spamreader:
			input_ = row[7]
			guess_ = row[6]
			ID = row[0]
			hmm_states = row[13]
			accuracy = row[22]
			bonus_round = row[29]
			Inputs.append(input_)
			Guesses.append(guess_)
			ParticipantIDs.append(ID)
			Input_Types.append(row[15])
			Hidden_states.append(hmm_states)
			Accuracy.append(accuracy)
			Bonus_round.append(bonus_round)
		Inputs.pop(0)
		Guesses.pop(0)
		ParticipantIDs.pop(0)
		Input_Types.pop(0)
		Hidden_states.pop(0)
		Accuracy.pop(0)
		Bonus_round.pop(0)

D_participant_l = []
R_participant_l = []
D_participant = []
R_participant = []
m = 5
for i in range(len(Inputs)):
	#foo = data['records'][i]
	if Input_Types[i]=='Clumpy':
		# load in the dictionary for that participant
		# figure out if they learned
		accuracy = Accuracy[i]
		output = Guesses[i]
		hidden_state = Hidden_states[i]
		accuracy = np.asarray([int(accuracy[i]) for i in range(len(accuracy))])
		output = np.asarray([int(output[i]) for i in range(len(output))])
		hidden_state = np.asarray([int(hidden_state[i]) for i in range(len(hidden_state))])
		T = len(accuracy)
		accuracy = accuracy[int(T/2):]
		output = output[int(T/2):]
		hidden_state = hidden_state[int(T/2):]
		D_participant.append(np.mean(accuracy))
		R_participant.append(rate(hidden_state,output))
		bonus_round = int(Bonus_round[i])
		if bonus_round>-1:
			# get the point on the curve
			#accuracy = accuracy[bonus_round-1:]
			#output = output[bonus_round-1:]
			#hidden_state = output[bonus_round-1:]
			# divide into parts and bootstrap
			foo_d = []; foo_r = []
			n = int(len(accuracy)/m)
			for j in range(m):
				accuracy_part = accuracy[j*n:(j+1)*n]
				output_part = output[j*n:(j+1)*n]
				hidden_state_part = hidden_state[j*n:(j+1)*n]
				d = np.mean(accuracy_part)
				r = rate(hidden_state_part,output_part)
				foo_d.append(d)
				foo_r.append(r)
			D_participant_l.append(foo_d)
			R_participant_l.append(foo_r)
		else:
			pass

betas = np.linspace(0,100,10000)
p = emissionProbs('EvenProcess')
Rs = []
Ds = []
for i in range(len(betas)):
	r, d = rate_distortion(p,betas[i],1e-9)
	Rs.append(r)
	Ds.append(d)

# np.savez('Clumpy_PRA.npz',Rs=Rs,Ds=Ds,R_participant=R_participant,
# 	D_participant=D_participant)

dat = np.load('Double_PRA.npz')
Rs = dat['Rs']; Ds = dat['Ds']
R_participant = dat['R_participant']
D_participant = dat['D_participant']

pl.rcParams.update({'text.usetex': True,
                     'font.family': 'sans-serif',
                     'font.sans-serif': "Helvetica"})
pl.plot(Rs,Ds,'-k',linewidth=1.5)
R_participant = np.asarray(R_participant)
D_participant = np.asarray(D_participant)
x = R_participant; #np.median(R_participant_l,1)
y = D_participant; #np.median(D_participant_l,1)
# xerrl = np.median(R_participant_l,1)-np.percentile(R_participant_l,16,1)
# xerr = np.std(R_participant_l,1)
# xerrh = np.percentile(R_participant_l,84,1)-np.median(R_participant_l,1)
# yerrl = np.median(D_participant_l,1)-np.percentile(D_participant_l,16,1)
# yerr = np.std(D_participant_l,1)
# yerrh = np.percentile(D_participant_l,84,1)-np.median(D_participant_l,1)
pl.plot(x,y,'ok')
pl.plot(np.max(Rs),np.max(Ds),'*r',markersize=15)
#pl.errorbar(x,y,np.vstack([yerrl,yerrh]),np.vstack([xerrl,xerrh]),'ok',ecolor=(0.5,0.5,0.5))
pl.xlabel('Rate (bits)',size=20)
pl.ylabel('Accuracy',size=20)
pl.savefig('Double_PRA2.pdf',bbox_inches='tight')
pl.show()

dat = np.load('EvenProcess_PRA.npz')
Rs = dat['Rs']; Ds = dat['Ds']
R_participant = dat['R_participant']
D_participant = dat['D_participant']

pl.rcParams.update({'text.usetex': True,
                     'font.family': 'sans-serif',
                     'font.sans-serif': "Helvetica"})
pl.plot(Rs,Ds,'-k',linewidth=1.5)
R_participant = np.asarray(R_participant)
D_participant = np.asarray(D_participant)
x = R_participant; #np.median(R_participant_l,1)
y = D_participant; #np.median(D_participant_l,1)
# xerrl = np.median(R_participant_l,1)-np.percentile(R_participant_l,16,1)
# xerr = np.std(R_participant_l,1)
# xerrh = np.percentile(R_participant_l,84,1)-np.median(R_participant_l,1)
# yerrl = np.median(D_participant_l,1)-np.percentile(D_participant_l,16,1)
# yerr = np.std(D_participant_l,1)
# yerrh = np.percentile(D_participant_l,84,1)-np.median(D_participant_l,1)
pl.plot(x,y,'ok')
pl.plot(np.max(Rs),np.max(Ds),'*r',markersize=15)
#pl.errorbar(x,y,np.vstack([yerrl,yerrh]),np.vstack([xerrl,xerrh]),'ok',ecolor=(0.5,0.5,0.5))
pl.xlabel('Rate (bits)',size=20)
pl.ylabel('Accuracy',size=20)
pl.savefig('EvenProcess_PRA2.pdf',bbox_inches='tight')
pl.show()

dat = np.load('NoisyPeriodic_PRA.npz')
Rs = dat['Rs']; Ds = dat['Ds']
R_participant = dat['R_participant']
D_participant = dat['D_participant']

pl.rcParams.update({'text.usetex': True,
                     'font.family': 'sans-serif',
                     'font.sans-serif': "Helvetica"})
pl.plot(Rs,Ds,'-k',linewidth=1.5)
R_participant = np.asarray(R_participant)
D_participant = np.asarray(D_participant)
x = R_participant; #np.median(R_participant_l,1)
y = D_participant; #np.median(D_participant_l,1)
# xerrl = np.median(R_participant_l,1)-np.percentile(R_participant_l,16,1)
# xerr = np.std(R_participant_l,1)
# xerrh = np.percentile(R_participant_l,84,1)-np.median(R_participant_l,1)
# yerrl = np.median(D_participant_l,1)-np.percentile(D_participant_l,16,1)
# yerr = np.std(D_participant_l,1)
# yerrh = np.percentile(D_participant_l,84,1)-np.median(D_participant_l,1)
pl.plot(x,y,'ok')
pl.plot(np.max(Rs),np.max(Ds),'*r',markersize=15)
#pl.errorbar(x,y,np.vstack([yerrl,yerrh]),np.vstack([xerrl,xerrh]),'ok',ecolor=(0.5,0.5,0.5))
pl.xlabel('Rate (bits)',size=20)
pl.ylabel('Accuracy',size=20)
pl.savefig('NoisyPeriodic_PRA2.pdf',bbox_inches='tight')
pl.show()

# make Vanessa's csv file
dat = np.load('Double_PRA.npz')
Rs = dat['Rs']; Ds = dat['Ds']
# interpolate
Rs_C = np.asarray([Rs[0]]); Ds_C = np.asarray([Ds[0]])
for i in range(len(Rs)-1):
	if Ds[i+1]>Ds[i]:
		r = np.linspace(Rs[i],Rs[i+1],100)
		d = (Ds[i+1]-Ds[i])/(Rs[i+1]-Rs[i])*r+Ds[i]-(Ds[i+1]-Ds[i])/(Rs[i+1]-Rs[i])*Rs[i]
		Rs_C = np.hstack([Rs_C,r])
		Ds_C = np.hstack([Ds_C,d])
	else:
		r = Rs[i]
		d = Ds[i]
		Rs_C = np.hstack([Rs_C,r])
		Ds_C = np.hstack([Ds_C,d])
Rs_C = np.asarray(Rs_C)
Ds_C = np.asarray(Ds_C)
# extend Rs to the maximal value of Rs
Rs_C = np.hstack([Rs_C,np.linspace(np.max(Rs_C),0.9542553220366964,100)])
Ds_C = np.hstack([Ds_C,np.ones(100)*np.max(Ds_C)])

dat = np.load('EvenProcess_PRA.npz')
Rs = dat['Rs']; Ds = dat['Ds']
# interpolate
Rs_E = np.asarray([Rs[0]]); Ds_E = np.asarray([Ds[0]])
for i in range(len(Rs)-1):
	if Ds[i+1]>Ds[i]:
		r = np.linspace(Rs[i],Rs[i+1],100)
		d = (Ds[i+1]-Ds[i])/(Rs[i+1]-Rs[i])*r+Ds[i]-(Ds[i+1]-Ds[i])/(Rs[i+1]-Rs[i])*Rs[i]
		Rs_E = np.hstack([Rs_E,r])
		Ds_E = np.hstack([Ds_E,d])
	else:
		r = Rs[i]
		d = Ds[i]
		Rs_E = np.hstack([Rs_E,r])
		Ds_E = np.hstack([Ds_E,d])
Rs_E = np.asarray(Rs_E)
Ds_E = np.asarray(Ds_E)

dat = np.load('NoisyPeriodic_PRA.npz')
Rs = dat['Rs']; Ds = dat['Ds']
# interpolate
Rs_N = np.asarray([Rs[0]]); Ds_N = np.asarray([Ds[0]])
for i in range(len(Rs)-1):
	if Ds[i+1]>Ds[i]:
		r = np.linspace(Rs[i],Rs[i+1],100)
		d = (Ds[i+1]-Ds[i])/(Rs[i+1]-Rs[i])*r+Ds[i]-(Ds[i+1]-Ds[i])/(Rs[i+1]-Rs[i])*Rs[i]
		Rs_N = np.hstack([Rs_N,r])
		Ds_N = np.hstack([Ds_N,d])
	else:
		r = Rs[i]
		d = Ds[i]
		Rs_N = np.hstack([Rs_N,r])
		Ds_N = np.hstack([Ds_N,d])
Rs_N = np.asarray(Rs_N)
Ds_N = np.asarray(Ds_N)

distances_orthogonal = []
distances_r = []
distances_a = []
for i in range(len(Inputs)):
	# get r and a
	n = len(Inputs[i])
	accuracy = np.asarray([int(k) for k in Accuracy[i][int(n/2):]])
	d = np.mean(accuracy)
	hidden = np.asarray([int(k) for k in Hidden_states[i][int(n/2):]])
	guess = np.asarray([int(k) for k in Guesses[i][int(n/2):]])
	r = rate(hidden,guess)
	if Input_Types[i]=='Clumpy':
		Rs = Rs_C; Ds = Ds_C
		entropy = 0.9542553220366964
	elif Input_Types[i]=='EvenProcess':
		Rs = Rs_E; Ds = Ds_E;
		p = np.array([10/13,3/13])
		entropy = np.min([1,-np.nansum(p*np.log2(p))])
	else:
		Rs = Rs_N; Ds = Ds_N;
		p = np.array([0.5,0.5])
		entropy = np.min([1,-np.nansum(p*np.log2(p))])
	# find the orthogonal distance
	deltaR = (Rs-r)/np.max(Rs)
	deltaD = (Ds-d)/np.max(Ds)
	distances_orthogonal.append(np.min(np.sqrt(deltaR**2+deltaD**2)))
	# find index for r
	ind_r = np.argmin(np.abs(Rs-r))
	distances_a.append((Ds[ind_r]-d)/Ds[ind_r])
	# find index for d
	ind_d = np.argmin(np.abs(Ds-d))
	distances_r.append((r-Rs[ind_d])/(entropy-Rs[ind_d]))

f = open('PRA_Distances','w')
writer = csv.writer(f)
writer.writerow(distances_orthogonal)
writer.writerow(distances_r)
writer.writerow(distances_a)
f.close()

distances_orthogonal_N = []; distances_orthogonal_C = []; distances_orthogonal_E = []
distances_r_N = []; distances_r_C = []; distances_r_E = []
distances_a_N = []; distances_a_C = []; distances_a_E = []
for i in range(len(Inputs)):
	if Input_Types[i]=='Clumpy':
		distances_r_C.append(distances_r[i])
		distances_a_C.append(distances_a[i])
		distances_orthogonal_C.append(distances_orthogonal[i]*np.sign(distances_a[i]))
	elif Input_Types[i]=='EvenProcess':
		distances_r_E.append(distances_r[i])
		distances_a_E.append(distances_a[i])
		distances_orthogonal_E.append(distances_orthogonal[i]*np.sign(distances_a[i]))
	else:
		distances_r_N.append(distances_r[i])
		distances_a_N.append(distances_a[i])
		distances_orthogonal_N.append(distances_orthogonal[i]*np.sign(distances_a[i]))

pl.figure()
g = sns.violinplot(data=[distances_orthogonal_N,distances_orthogonal_C,distances_orthogonal_E],
	palette=['tab:blue', 'tab:orange', 'tab:green'],color=0.5)
sns.stripplot(data=[distances_orthogonal_N,distances_orthogonal_C,distances_orthogonal_E],
	edgecolor='k',linewidth=1,alpha=0.3,jitter=True)
g.set_xticklabels(['Noisy Periodic','Double Process','Even Process'])
pl.savefig('PRADistances_orthogonal.pdf',bbox_inches='tight')
pl.show()

pl.figure()
g = sns.violinplot(data=[distances_r_N,distances_r_C,distances_r_E],
	palette=['tab:blue', 'tab:orange', 'tab:green'],color=0.5)
sns.stripplot(data=[distances_r_N,distances_r_C,distances_r_E],
	edgecolor='k',linewidth=1,alpha=0.3,jitter=True)
g.set_xticklabels(['NoisyPeriodic','Double Process','Even Process'])
pl.savefig('PRADistances_rate.pdf',bbox_inches='tight')
pl.show()

pl.figure()
g = sns.violinplot(data=[distances_a_N,distances_a_C,distances_a_E],
	palette=['tab:blue', 'tab:orange', 'tab:green'],color=0.5)
sns.stripplot(data=[distances_a_N,distances_a_C,distances_a_E],
	edgecolor='k',linewidth=1,alpha=0.3,jitter=True)
g.set_xticklabels(['NoisyPeriodic','Double Process','Even Process'])
pl.savefig('PRADistances_accuracy.pdf',bbox_inches='tight')
pl.show()