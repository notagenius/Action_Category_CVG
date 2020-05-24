import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import matplotlib.animation as animation

with open('labeled_json.json', 'r') as f:
    data = json.load(f)

metadata = data['metadata']

label_rd = []

for k in metadata:
	single_label = metadata[k]
	av = single_label['av']
	label = av['1']
	z = single_label['z']
	start_ts = int(z[0] * 50)
	end_ts = int(z[1] * 50)
	label_rd.append([label,start_ts,end_ts])


def plot(ax, human, connect, LR, lcolor, rcolor, T=np.zeros((3, )),
         linewidth=3, alpha=0.8, plot_jid=False, do_scatter=False):
    """
    :param ax: matplotlib subplot
    :param human: [J x 3]
    :param connect: [(jid1, jid2), (jid1, jid2), ...  ]
    :param LR: [True, True, False, ...] 
        define if jid is left or right
    """
    human = human + T
    for a, b in connect:
        if isinstance(a, int):
            A = human[a]
        else:
            assert len(a) == 2
            left, right = a
            A = (human[left] + human[right]) / 2
            a = b  # make it integer
        
        B = human[b]
        is_left = LR[a] or LR[b]
        color = lcolor if is_left else rcolor

        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]],
                color=color, alpha=alpha, linewidth=linewidth)
        
        
    if do_scatter:
        ax.scatter(human[:, 0], human[:, 1], human[:, 2],
                   color='gray', alpha=0.4)
    
    if plot_jid:
        for i, (x, y, z) in enumerate(human):
            ax.text(x, y, z, str(i))

def plot_pts3d(ax, Pts, lcolor='blue', rcolor='red', plot_jid=False, T=np.zeros((3, ))):
    if len(Pts.shape) == 1:
        assert len(Pts) == 96, str(Pts.shape)
        Pts = Pts.reshape((-1, 3))
    elif len(Pts.shape) == 2:
        J, dim = Pts.shape
        assert J == 32, str(J)
        assert dim == 3, str(dim)
    else:
        raise NotImplementedError
    
    connect = [
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (0, 1), (0, 6),
        (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (24, 25), (24, 17),
        (24, 14), (14, 15)
    ]
    LR = [
        False, True, True, True, True,
        True, False, False, False, False,
        False, True, True, True, True,
        True, True, False, False, False,
        False, False, False, False, True,
        False, True, True, True, True,
        True, True
    ]
    Pts[:, (0, 1, 2)] = Pts[:, (0, 2, 1)]

    plot(ax, Pts, connect, LR=LR, lcolor=lcolor, rcolor=rcolor,
            plot_jid=plot_jid, do_scatter=False, T=T)


input_array = np.load("pose3d/S1_directions_1.npy")

input_array = input_array[::5,:,:]
fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

input_plot = plot_pts3d(ax, input_array[0])

def my_update(number):
    ax.clear()
    s="nothing"
    input_plot = plot_pts3d(ax, input_array[number])
    for i in label_rd:
        if number*5 >= i[1] and number*5 <= i[2]:
            s=i[0]
    ax.text(0.9, 0.9, 0.9, s, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=25, color='black')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    return input_plot,


animation = animation.FuncAnimation(fig, my_update, frames=len(input_array),interval=1)
plt.show()
