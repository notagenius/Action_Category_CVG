import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

def vid_idx2actor(vid_idx, data):
	filename = data["file"][vid_idx]['fname'].strip().split("/")
	actor = filename[8].split("_")[1]
	return actor

def timestamp2frame(time_stamp):
    return round(time_stamp * 50)

def timestamp_pair2frame_pair(input):
	return int(timestamp2frame(input[0])), int(timestamp2frame(input[1]))

def interplation():
    return
	

def label2int(label_list):
	label_list = list(set(label_list))
	label_list.sort()
	print(label_list)
	idx = 0
	label_dict = {} 
	for i in label_list:
		label_dict[i] = idx
		idx = idx + 1
	return label_dict

def swipe_dict(in_dict):
	return dict((v,k) for k,v in in_dict.items())

with open('via_latest.json') as json_file:
	data = json.load(json_file)
	vid_idx = []
	label = []
	actor = []
	pair = []
	time = []
	for i in data['metadata']:
		vid_idx.append(data['metadata'][i]['vid'])
		actor.append(vid_idx2actor(data['metadata'][i]['vid'], data).strip('S'))
		label.append(data['metadata'][i]['av']['1'])
		time.append(timestamp_pair2frame_pair(data['metadata'][i]['z']))
		pair.append([data['metadata'][i]['vid'], vid_idx2actor(data['metadata'][i]['vid'], data).strip('S'), data['metadata'][i]['av']['1'], int(timestamp_pair2frame_pair(data['metadata'][i]['z'])[0]), int(timestamp_pair2frame_pair(data['metadata'][i]['z'])[1])])
	#print(pair)

label_dict = label2int(label)
dict_label = swipe_dict(label_dict)

video = {}
for i in pair:
	if (i[0] in video):
		ele = [i[1],label_dict[i[2]],i[3],i[4]]
		video[i[0]].append(ele)
	else:
		ele = [i[1],label_dict[i[2]],i[3],i[4]]
		video[i[0]] = [ele]

print(video[u'70'])
# one video example

# mtype = np.dtype('np.int32, np.int32, np.int32, np.int32')
v70 = np.asarray(video[u'70'], dtype = ('int32','int32','int32','int32'))

print(np.sort(v70,axis=2))


# sorting


'''
# selecting the most labelled video
result = []
for j in video:
	if video[j] == 23:
		print(j)

'''




'''
actor = list(set(actor))
actor = ["1","11","5","6","7","8","9"]

label = list(set(label))

accumulation = np.zeros(shape=(len(actor),len(label)))

label_a = dict(zip(actor, range(0,len(actor))))
label_d = dict(zip(label, range(0,len(label))))
res_label_d = dict((v,k) for k,v in label_d.items())

for i in actor:
	for j in label:
		count = 0 
		for k in pair:
			if k[0] == i and k[1] == j:
				count = count + 1
		accumulation[int(label_a[i])][int(label_d[j])] = count

for i in range(0,accumulation.shape[0]):
	ran = random.uniform(0.5, 1.9)
	for j in range(0,accumulation.shape[1]):
		ran2 = random.uniform(0.8, 1.2)
		if accumulation[i][j] <= 0.0:
			accumulation[i][j] = int(accumulation[int(label_a["11"])][j] * ran * ran2)
'''
