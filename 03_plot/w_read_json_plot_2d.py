import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

def vid_idx2actor(vid_idx, data):
	filename = data["file"][vid_idx]['fname'].strip().split("/")
	actor = filename[8].split("_")[1]
	return actor
	

with open('via_latest.json') as json_file:
	data = json.load(json_file)
	vid_idx = []
	label = []
	actor = []
	pair = []
	for i in data['metadata']:
			vid_idx.append(data['metadata'][i]['vid'])
			actor.append(vid_idx2actor(data['metadata'][i]['vid'], data).strip('S'))
			label.append(data['metadata'][i]['av']['1'])
			pair.append([vid_idx2actor(data['metadata'][i]['vid'], data).strip('S'), data['metadata'][i]['av']['1']])



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

#print(accumulation)

# for plot
 
# set width of bar
barWidth = 0.1
 
# set height of bar
bars0 = accumulation[0]
bars1 = accumulation[1]
bars2 = accumulation[2]
bars3 = accumulation[3]
bars4 = accumulation[4]
bars5 = accumulation[5]
bars6 = accumulation[6]


 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]

 
# Make the plot
plt.bar(r1, bars0, color='#7f6d5f', width=barWidth, edgecolor='white', label="Actor"+actor[0])
plt.bar(r2, bars1, color='#557f2d', width=barWidth, edgecolor='white', label="Actor"+actor[1])
plt.bar(r3, bars2, color='#404140', width=barWidth, edgecolor='white', label="Actor"+actor[2])
plt.bar(r4, bars3, color='#597452', width=barWidth, edgecolor='white', label="Actor"+actor[3])
plt.bar(r5, bars4, color='#2d7f5e', width=barWidth, edgecolor='white', label="Actor"+actor[4])
plt.bar(r6, bars5, color='#819799', width=barWidth, edgecolor='white', label="Actor"+actor[5])
plt.bar(r7, bars6, color='#C7C3A7', width=barWidth, edgecolor='white', label="Actor"+actor[6])

 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], [res_label_d[0], res_label_d[1], res_label_d[2], res_label_d[3], res_label_d[4], res_label_d[5], res_label_d[6], res_label_d[7], res_label_d[8], res_label_d[9]])
 
# Create legend & Show graphic
plt.legend()
plt.show()


plt.savefig('2d.eps', format='eps')
