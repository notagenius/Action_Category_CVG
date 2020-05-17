from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import numpy as np
import glob
from pandas import Series, DataFrame
from torch import nn
from tensorboardX import SummaryWriter
import csv
import pandas as pd

writer = SummaryWriter("baseline_dir")
params = { 'batch_size': 64, 'shuffle': True, 'num_workers': 10, 'drop_last': True}

def get_nm_typ(file):
    filename = file.split("/")[-1].split('.')[:-1]
    file_type = file.split("/")[-1].split('.')[-1]
    return filename, file_type

class my_dataset_2(Dataset):
    def __init__(self, csv_path_folder, npy_path_folder):
        self.csv_filenames = sorted(glob.glob(csv_path_folder))
        self.csv_list_of_dfs = [pd.read_csv(filename) for filename in self.csv_filenames]
        print(self.csv_list_of_dfs[0])

class my_dataset(Dataset):
    def __init__(self, csv_path_folder, npy_path_folder):
        self.csv_fnames = sorted(glob.glob(csv_path_folder))
        self.csv_ls = []
        for f in self.csv_fnames:
            with open(f) as csvfile:
                data = csv.DictReader(csvfile)
                self.csv_ls.append(data)
        self.csv_dict = {}
        self.csv_fname = []
        for csv_df, csv_fn in zip(self.csv_ls, self.csv_fnames):
            tmp_name, _ = get_nm_typ(csv_fn)
            self.csv_fname.append(tmp_name)
            self.csv_dict[tmp_name] = csv_df
            

csv_path = {'train':"../label_gt/csv/01_minial_scope_label/S*.csv", 'val':"../label_gt/csv/01_minial_scope_label_val/S*.csv"}
npy_path = {'train':"../label_gt/pose3d/S*.npy",'val':"../label_gt/pose3d_val/S*.npy"}

training_set = my_dataset(csv_path['train'], npy_path['train'])