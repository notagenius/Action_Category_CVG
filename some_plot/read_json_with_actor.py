import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

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
			pair.append([vid_idx2actor(data['metadata'][i]['vid'], data), data['metadata'][i]['av']['1']])

actor = list(set(actor))
label = list(set(label))

accumulation = []

for i in actor:
	for j in label:
		count = 0 
		for k in pair:
			if k[0] == i and k[1] == j:
				count = count + 1
		accumulation.append([i,j,count])

#print(accumulation)
print(label)

# for plot

np.random.seed(19680801)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']

yticks = list(map(int, actor))

for c, k in zip(colors, yticks):
    # Generate the random data for the y=k 'layer'.
    xs = label
    ys = np.random.rand(20)

    # You can provide either a single color or an array with the same length as
    # xs and ys. To demonstrate this, we color the first bar of each set cyan.
    #cs = [c] * len(xs)

    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    #ax.bar(label, ys, zs=k, zdir='y', color=cs, alpha=0.8)
    ax.bar(label, ys, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# On the y axis let's only label the discrete values that we have data for.
ax.set_yticks(yticks)

plt.show()	
