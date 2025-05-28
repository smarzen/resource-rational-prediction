import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib

files = ['NoisyPeriodicData.txt',
'EvenProcessData.txt',
'DoubleData.txt']

np_data = {}
ep_data = {}
cl_data = {}

# parsing data files into dictionaries 

def fill_dictionary(file_path,dict):
    f = open(file_path)
    lines = f.readlines()
    lines = [l for l in lines if l != '\n']
    id = ''
    for line in lines:
        if ')' in line:
            id = line[:12]
            if id not in dict.keys():
                dict[id] = {}
        elif line[0] == 'n':
            unassigned = True
            n = 0
            while unassigned == True:
                if 'n-gram' in dict[id].keys():
                      n += 1
                      id = id[:12] + f'({n})'
                      if id not in dict.keys():
                        dict[id] = {}
                else:
                    dict[id]['n-gram'] = float(line[11:-1])
                    unassigned = False                
        elif line[0] == 'B':
            dict[id]['BSI'] = float(line[5:-1])
        elif line[0] == 'G':
            dict[id]['GLM'] = float(line[5:-1])
        elif line[0] == 'l':
            unassigned = True
            n = 0
            while unassigned == True:
                if 'LSTM' in dict[id].keys():
                      n += 1
                      id = id[:12] + f'({n})'
                      if id not in dict.keys():
                        dict[id] = {}
                else:
                    dict[id]['LSTM'] = float(line[6:-1])
                    unassigned = False 
            dict[id]['LSTM'] = float(line[6:-1])
    f.close()
    return

def update_LSTM(file_path, dict): 
    f = open(file_path)
    lines = f.readlines()
    lines = [l for l in lines if l != '\n']
    id = ''
    tracked_ids = []
    for line in lines:
        if ')' in line:
            id = line[:12]
            if id not in dict.keys():
                print("found unique id: ", id)
        elif line[0] == 'l':
            id_found = False
            n = 0
            while id_found == False:
                if id in tracked_ids:
                      n += 1
                      id = id[:12] + f'({n})'
                      if id not in dict.keys():
                        print("found unique id: ", id)
                else:
                    if dict[id]['LSTM'] > float(line[6:-1]):
                        dict[id]['LSTM'] = float(line[6:-1])
                    id_found = True
    f.close()
    return

# counting which models best fit each entry

def count(dict):
    new_dict = {}
    # list of best candidates
    ans = []
    # dictionary of best candidates corresponding to participant id
    # result = {}
    for id in dict.keys():
        m = max(dict[id], key = dict[id].get)
        ans.append(m)
        # result[id] = m
    # counting results into new dictionary 
    new_dict['n-gram'] = ans.count('n-gram')
    new_dict['BSI'] = ans.count('BSI')
    new_dict['GLM'] = ans.count('GLM')
    new_dict['LSTM'] = ans.count('LSTM')
    return new_dict 

# initializing data
fill_dictionary(files[0],np_data)
fill_dictionary(files[1],ep_data)
fill_dictionary(files[2],cl_data)

counts = [count(np_data),count(ep_data),count(cl_data)]
print(counts)

# updating with best lstm scores
for i in range(2,5):
    path = f'NoisyPeriodicData{i}.txt'
    update_LSTM(path, np_data)
    path = f'EvenProcessData{i}.txt'
    update_LSTM(path, ep_data)
    path = f'ClumpyData{i}.txt'
    update_LSTM(path, cl_data)

counts = [count(np_data),count(ep_data),count(cl_data)]
print(counts)

# pandas method
colors = ["#e8bcd3","#f5e4b8","#b0c5d9","#b1d9b0"]
dict = {'Labels': ['NoisyPeriodic','EvenProcess','Double'],
        'Values': [count(np_data),count(ep_data),count(cl_data)]}
df = pd.DataFrame(dict['Values'], index=dict['Labels'])
df.plot(kind="bar", stacked=True, color = colors)

plt.xticks(rotation=0)
plt.rcParams.update({'text.usetex': True,
                     'font.family': 'sans-serif',
                     'font.sans-serif': "Helvetica"})
plt.show()



# counts = [count(np_data),count(ep_data),count(cl_data)]
# print([counts[i]['ngram'] for i in range(3)])
# labels = ['NoisyPeriodic','EvenProcess','Clumpy']
# ngram_vals = [counts[i]['ngram'] for i in range(3)]
# bsi_vals = [counts[i]['bsi'] for i in range(3)]
# glm_vals = [counts[i]['glm'] for i in range(3)]
# lstm_vals = [counts[i]['lstm'] for i in range(3)]
# fig, ax = plt.subplots()
# ax.bar(labels, ngram_vals, color = "#e8bcd3") # pink
# ax.bar(labels, bsi_vals, color = "#f5e4b8") # yellow
# ax.bar(labels, glm_vals, color = "#b0c5d9") # blue
# ax.bar(labels, lstm_vals, color = "#b1d9b0") # green