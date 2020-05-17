import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib


def rstyle(ax):
    """Styles an axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been carried out (needs to know final tick spacing)
    """
    #set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='0.4', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.4', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.5')
    ax.set_axisbelow(True)
   
    #set minor tick spacing to 1/2 of the major ticks
    #ax.xaxis.set_minor_locator(MultipleLocator( (plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 ))
    ax.yaxis.set_minor_locator(MultipleLocator( (plt.yticks()[0][1]-plt.yticks()[0][0]) / 2.0 ))
   
    #remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)
       
    #restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)
   
    #remove the minor tick lines    
    for line in ax.xaxis.get_ticklines(minor=True) + ax.yaxis.get_ticklines(minor=True):
        line.set_markersize(0)
   
    #only show bottom left ticks, pointing out of axis
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
   
   
    if ax.legend_ != None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)
        

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
 
plt.style.use('ggplot')

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

fig = plt.figure()
ax = fig.add_subplot(111)
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]

 
# Make the plot
plt.bar(r1, bars0, color='#ba9b6b',alpha = 0.65, width=barWidth, edgecolor='white', label="Actor"+actor[0])
plt.bar(r2, bars1, color='#abaa8c',alpha = 0.65, width=barWidth, edgecolor='white', label="Actor"+actor[1])
plt.bar(r3, bars2, color='#789b83',alpha = 0.65, width=barWidth, edgecolor='white', label="Actor"+actor[2])
plt.bar(r4, bars3, color='#ba6557',alpha = 0.65, width=barWidth, edgecolor='white', label="Actor"+actor[3])
plt.bar(r5, bars4, color='#87502f',alpha = 0.65, width=barWidth, edgecolor='white', label="Actor"+actor[4])
plt.bar(r6, bars5, color='#9eaeb9',alpha = 0.65, width=barWidth, edgecolor='white', label="Actor"+actor[5])
plt.bar(r7, bars6, color='#7c2a11',alpha = 0.65, width=barWidth, edgecolor='white', label="Actor"+actor[6])

 
# Add xticks on the middle of the group bars
plt.xlabel('Weiling Styling Matplotlib Sample', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], [res_label_d[0], res_label_d[1], res_label_d[2], res_label_d[3], res_label_d[4], res_label_d[5], res_label_d[6], res_label_d[7], res_label_d[8], res_label_d[9]])

rstyle(ax)
# Create legend & Show graphic
plt.legend()
plt.show()


plt.show()	
