from sklearn.utils import class_weight
import numpy as np

re =[]
f = open("train.weight", "r")
for x in f:
  re.append(x.strip('\n'))

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(re),
                                                 re)

print(class_weights)
