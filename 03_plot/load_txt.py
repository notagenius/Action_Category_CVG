import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

f = open("result2plt.txt", "r")
lines = f.readlines()

count = []
label = []
for i in lines:
	line=i.strip().split(' ')
	count.append(int(line[0]))
	label.append(line[1])

print(count)
print(label)

fig, ax = plt.subplots()

y_pos = label

ax.barh(label, count, align='center')
ax.set_ylabel('label')
#ax.set_yticks(label)
#ax.set_yticklabels(label)
#ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('count')
ax.set_title('label analysis')

plt.show()

plt.savefig(os.path.join('overall_label.png'), dpi=300, format='png', bbox_inches='tight')

#fig = go.Figure(data=[go.Histogram(y=count)])
#fig.show()
