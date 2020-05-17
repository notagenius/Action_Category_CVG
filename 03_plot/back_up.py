import json
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

def rstyle(ax):
    """Styles an axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been carried out (needs to know final tick spacing)
    """
    #set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.85')
    ax.set_axisbelow(True)
   
    #set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator(MultipleLocator( (plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 ))
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
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
   
   
    if ax.legend_ != None:
        lg = ax.legend_
        lg.get_frame().set_linewidth(0)
        lg.get_frame().set_alpha(0.5)
       
       
def rhist(ax, data, **keywords):
    """Creates a histogram with default style parameters to look like ggplot2
    Is equivalent to calling ax.hist and accepts the same keyword parameters.
    If style parameters are explicitly defined, they will not be overwritten
    """
   
    defaults = {
                'facecolor' : '0.3',
                'edgecolor' : '0.28',
                'linewidth' : '1',
                'bins' : 100
                }
   
    for k, v in defaults.items():
        if k not in keywords: keywords[k] = v
   
    return ax.hist(data, **keywords)


def rbox(ax, data, **keywords):
    """Creates a ggplot2 style boxplot, is eqivalent to calling ax.boxplot with the following additions:
   
    Keyword arguments:
    colors -- array-like collection of colours for box fills
    names -- array-like collection of box names which are passed on as tick labels

    """

    hasColors = 'colors' in keywords
    if hasColors:
        colors = keywords['colors']
        keywords.pop('colors')
       
    if 'names' in keywords:
        ax.tickNames = plt.setp(ax, xticklabels=keywords['names'] )
        keywords.pop('names')
   
    bp = ax.boxplot(data, **keywords)
    pylab.setp(bp['boxes'], color='black')
    pylab.setp(bp['whiskers'], color='black', linestyle = 'solid')
    pylab.setp(bp['fliers'], color='black', alpha = 0.9, marker= 'o', markersize = 3)
    pylab.setp(bp['medians'], color='black')
   
    numBoxes = len(data)
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
          boxX.append(box.get_xdata()[j])
          boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX,boxY)
       
        if hasColors:
            boxPolygon = Polygon(boxCoords, facecolor = colors[i % len(colors)])
        else:
            boxPolygon = Polygon(boxCoords, facecolor = '0.95')
           
        ax.add_patch(boxPolygon)
    return bp


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
rstyle(ax)
plt.show()
