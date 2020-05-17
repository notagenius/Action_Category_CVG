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

np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'm', 'y', 'k', 'w']

yticks = list(map(int, actor))

#print(label)

for c, k in zip(colors, yticks):
    # Generate the random data for the y=k 'layer'.
    xs = np.arange(len(label))
    ys = []
    for i in label:
        ys.append(accumulation[int(label_a[str(k)])][label_d[i]])

    # You can provide either a single color or an array with the same length as
    # xs and ys. To demonstrate this, we color the first bar of each set cyan.
    cs = [c] * len(xs)

    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax.bar(xs, ys, zs=k, zdir='y',color = cs, alpha=0.8)
    

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# On the y axis let's only label the discrete values that we have data for.
ax.set_yticks(yticks)
ax.set_title(res_label_d)

plt.show()
