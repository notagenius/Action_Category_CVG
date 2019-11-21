from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# for csv parsing
import pandas as pd
import numpy as np
# fro folder
import glob

def flatten_dict(in_dict):
    out 

class my_dataset_from_csv:
    def __init__(self, csv_path_folder):
        filenames = glob.glob(csv_path_folder)
        list_of_dfs = [pd.read_csv(filename) for filename in filenames]
        dataframes = {}
        for dataframe, filename in zip(list_of_dfs, filenames):
            dataframes[filename] = dataframe
        print(len(list_of_dfs))
        print(len(dataframes))
        conbined_df = pd.concat(list_of_dfs, ignore_index=True)
        print(len(conbined_df))
        # to tensor
        #self.to_tensor = transforms.ToTensor()

if __name__ == "__main__":
    my_dataset_from_csv("../label_gt/csv/01_minial_scope_label/S*.csv")



#def __getitem__(self, index):
#    img_path, label = 
