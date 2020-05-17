import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

def vid_idx2actor(vid_idx, data):
	filename = data["file"][vid_idx]['fname'].strip().split("/")
	actor = filename[8].split("_")[1]
	return actor

def vid_idx2filename(vid_idx, data):
	filename = data["file"][vid_idx]['fname'].strip().split("/")
	filename = filename[8].split("_")
	filename[2] = filename[2].lower()
	if filename[2] == "takingphoto":
		filename[2] = "photo"
	if filename[2] == "walktogether":
		filename[2] = "walkingtogether"
	if filename[2] == "walkingdog":
		filename[2] = "walkdog"
	filename = "_".join(filename[1:4]) + ".csv"
	return filename

def read_frame_nr_of_pose3d(csvfile):
	gt_3d = np.load('../00_datasets/pose3d/'+csvfile[:-3]+"npy")
	return gt_3d.shape[0]

def timestamp2frame(time_stamp):
    return round(time_stamp * 50)

def timestamp_pair2frame_pair(input):
	return int(timestamp2frame(input[0])), int(timestamp2frame(input[1]))

def interplation(in_list):
	result = []
	for index,item in enumerate(in_list):
		if index == 0 and len(in_list) == 1:
			tmp = list(item)
			if item[2] - 0 < 10:
				tmp[2] = 0
			else:
				print("!!!! something is wrong")
				print item
		elif index == 0 and len(in_list)!= 1:
			tmp = list(item)
			if item[2] - 0 < 10:
				tmp[2] = 0
			tmp[3] = int(round((item[3] + in_list[index+1][2])/2))
		elif index ==  len(in_list) - 1:
			tmp = list(item)
			tmp[2] = int(round((item[2] + in_list[index-1][3])/2))
		else:
			tmp = list(item)
			tmp[2] = int(round((item[2] + in_list[index-1][3])/2))
			tmp[3] = int(round((item[3] + in_list[index+1][2])/2))
		result.append(tmp)
	return result

def sort_label_along_frame(in_list):
	dict = {}
	for i in in_list:
		dict[i[2]] = i
	keys = dict.keys()
	keys.sort()
	return [dict[key] for key in keys]
	
def label2int(label_list):
	label_list = list(set(label_list))
	label_list.sort()
	idx = 0
	label_dict = {} 
	for i in label_list:
		label_dict[i] = idx
		idx = idx + 1
	return label_dict

def cut_label_list(in_list, cut_frame):
	result = []
	for i in in_list:
		if i[2] < cut_frame and i[3] < cut_frame:
			result.append(i)
		if i[2] < cut_frame and i[3] >= cut_frame:
			i[3] = cut_frame
			result.append(i)
			return result


def write_csv(csvfile, in_list, label_set):
	diff = int(in_list[-1][3]) - int(read_frame_nr_of_pose3d(csvfile))
	if csvfile == "S9_greeting_2.csv" or csvfile =="S9_phoning_2.csv" or csvfile =="S9_sittingdown_1.csv" or csvfile =="S9_waiting_1.csv" or csvfile =="S9_walking_2.csv":
		in_list = cut_label_list(in_list, read_frame_nr_of_pose3d(csvfile))
	else:
		in_list[-1][3] = read_frame_nr_of_pose3d(csvfile)
	f = open(csvfile, "w")
	for i in in_list:
		listofzeros = ["0"] * len(label_set)
		line = i[3] - i[2]
		for w in range(line):
			listofzeros[i[1]] = "1"
			f.write(",".join(listofzeros))
			f.write('\n')
	f.close()
	return

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


label_dict = label2int(label)
dict_label = swipe_dict(label_dict)

print(label_dict)

video = {}


for i in pair:
	if (i[0] in video):
		ele = [i[1],label_dict[i[2]],i[3],i[4]]
		video[i[0]].append(ele)
	else:
		ele = i[1],label_dict[i[2]],i[3],i[4]
		video[i[0]] = [ele]

for i in pair:
	test=interplation(sort_label_along_frame(video[i[0]]))
	filename = vid_idx2filename(i[0],data)
	#if filename == "S11_phoning_2.csv":
	#	filename = "S11_phoning_1.csv"
	if filename == "S5_discussion_2.csv":
		filename = "S5_discussion_1.csv"
	if filename == "S6_waiting_2.csv":
		filename = "S6_waiting_1.csv"
	if filename == "S11_phoning_3.csv":
		filename = "S11_phoning_1.csv"
	if filename == "S5_discussion_3.csv":
		filename = "S5_discussion_2.csv"
	if filename == "S6_waiting_3.csv":
		filename = "S6_waiting_2.csv"
	if filename == "S5_photo_11oclock.mp4.csv":
		filename = "S5_photo_1.csv"


	write_csv(filename, test, label_dict)



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
