from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from torch import nn
import argparse
import glob
from tensorboardX import SummaryWriter
import pandas as pd


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', type=str, default='pose', help='task to be trained')
    parser.add_argument('-r', '--runs', type=str, default='test/baseline-001', help='tensorboard location')
    parser.add_argument('-b', '--batchsize', type=int, default=64, help='batchsize')
    #parser.add_argument('-l', '--force_learning_rate', type=float, default=-1.0, help='setting learning rate')
    args = parser.parse_args()
    return args

params = { 'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': True}


def default_loader(csv_path_folder, npy_path_folder):
    return

def get_filename_type(file):
    filename = file.split("/")[-1].split('.')[:-1]
    file_type = file.split("/")[-1].split('.')[-1]
    return filename, file_type


class my_dataset2(Dataset):
    def __init__(self, csv_path_folder, npy_path_folder):
        self.csv_filenames = sorted(glob.glob(csv_path_folder))
        self.csv_list_of_dfs = [pd.read_csv(filename) for filename in self.csv_filenames]
        self.csv_dataframes = {}
        self.csv_filename = []
        for csv_dataframe, csv_filename in zip(self.csv_list_of_dfs, self.csv_filenames):
            tmp_name,_= get_filename_type(csv_filename)
            self.csv_filename.append(tmp_name)
            self.csv_dataframes[csv_filename] = csv_dataframe
        self.csv_conbined_df = pd.concat(self.csv_list_of_dfs, ignore_index=True)
        self.csv_torch_tensor = torch.tensor(self.csv_conbined_df.values)
        self.npy_filenames = sorted(glob.glob(npy_path_folder))
        self.npy_list_of_frames = [np.load(filename) for filename in self.npy_filenames]
        self.npy_inputs = {}
        self.npy_filename = []

        for npy_input, npy_filename in zip(self.npy_list_of_frames, self.npy_filenames):
            tmp_name,_= get_filename_type(npy_filename)
            if tmp_name not in self.csv_filename:
                self.npy_inputs[npy_filename] = npy_input
        self.npy_conbined_inputs = np.concatenate(self.npy_list_of_frames, axis=0, out=None)
        self.npy_torch_tensor = torch.tensor(self.npy_conbined_inputs)
        print(self.csv_conbined_df.values[1])
        print("length of input skeleton is:"+str(len(self.npy_conbined_inputs))+" mod of batch size is:"+str(len(self.npy_conbined_inputs)%params['batch_size']))
        print("length of input label is:"+str(len(self.csv_conbined_df.values))+" mod of batch size is:"+str(len(self.csv_conbined_df.values)%params['batch_size']))

    def __len__(self):
        return len(self.csv_conbined_df)

    def __getitem__(self, index):
        return np.concatenate(self.npy_conbined_inputs[index]),self.csv_conbined_df.values[index].argmax(axis=0)


class my_dataset(Dataset):
    def __init__(self, csv_path_folder, npy_path_folder):
        self.csv_filenames = sorted(glob.glob(csv_path_folder))
        self.csv_list_of_dfs = [np.loadtxt(filename, dtype=np.float32) for filename in self.csv_filenames]
        self.csv_dataframes = {}
        self.csv_filename = []
        for csv_dataframe, csv_filename in zip(self.csv_list_of_dfs, self.csv_filenames):
            tmp_name,_= get_filename_type(csv_filename)
            self.csv_filename.append(tmp_name)
            self.csv_dataframes[csv_filename] = csv_dataframe
        self.csv_conbined_df = np.concatenate(self.csv_list_of_dfs)
        self.csv_torch_tensor = torch.tensor(self.csv_conbined_df)
        self.npy_filenames = sorted(glob.glob(npy_path_folder))
        self.npy_list_of_frames = [np.load(filename) for filename in self.npy_filenames]
        self.npy_inputs = {}
        self.npy_filename = []

        for npy_input, npy_filename in zip(self.npy_list_of_frames, self.npy_filenames):
            tmp_name,_= get_filename_type(npy_filename)
            if tmp_name not in self.csv_filename:
                self.npy_inputs[npy_filename] = npy_input
        self.npy_conbined_inputs = np.concatenate(self.npy_list_of_frames, axis=0, out=None)
        self.npy_torch_tensor = torch.tensor(self.npy_conbined_inputs)
        print(self.csv_conbined_df[1])

        print("length of input skeleton is:"+str(len(self.npy_conbined_inputs))+" mod of batch size is:"+str(len(self.npy_conbined_inputs)%params['batch_size']))
        print("length of input label is:"+str(len(self.csv_conbined_df))+" mod of batch size is:"+str(len(self.csv_conbined_df)%params['batch_size']))

    def __len__(self):
        return len(self.csv_conbined_df)

    def __getitem__(self, index):
        return np.concatenate(self.npy_conbined_inputs[index]),self.csv_conbined_df[index].argmax(axis=0)

class FCL_no_activate(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCL_no_activate, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class FCL(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FCL, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class simple_FCN(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simple_FCN, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Activation_FCN(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_FCN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Batch_FCN(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_FCN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=0)
    return nn.CrossEntropyLoss()(input, labels)

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(use_cuda)

    opt = parse_command_line()
    writer = SummaryWriter(opt.runs)


    max_epochs = 100

    csv_path = {'val':"../00_datasets/Julian_data/label_not5/S*.txt", 'train':"../00_datasets/Julian_data/label_5/S*.txt"}
    npy_path = {'val':"../00_datasets/Weiling_data/pose_not5/S*.npy",'train':"../00_datasets/Weiling_data/pose_5/S*.npy"}

    #csv_path = {'val':"../00_datasets/Weiling_data/label_not5/S*.csv", 'train':"../00_datasets/Weiling_data/label_5/S*.csv"}
    #npy_path = {'val':"../00_datasets/Weiling_data/pose_not5/S*.npy",'train':"../00_datasets/Weiling_data/pose_5/S*.npy"}


    training_set = my_dataset(csv_path['train'], npy_path['train'])
    training_generator = DataLoader(training_set, **params)

    validation_set = my_dataset(csv_path['val'], npy_path['val'])
    validation_generator = DataLoader(validation_set, **params) 

    need_print = True
    need_print_tail = True

    batch_size = params['batch_size']
    input_size = 32 * 3 
    hidden_size = 512 
    num_classes = 11
    num_epochs = 1
    total_step = 200
    
    learning_rate = 0.0001
    #model = FCL(input_size, hidden_size, num_classes).to(device)
    model = Activation_FCN(input_size, hidden_size, hidden_size*2, num_classes).to(device)
    #model = FCL_no_activate(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    count = 0
    best_accu = 0
    for epoch in range(max_epochs):

        #Validation
        with torch.set_grad_enabled(False):
            PATH = "/media/data/weiling/Action_Category_CVG/model/"+"best.pth"
            model.load_state_dict(torch.load(PATH))
            model.eval()
            for skeleton_val, label_val in validation_generator:
                skeleton, label = skeleton_val.to(device), label_val.to(device)
            #skeletons = skeleton.reshape(-1, 32 * 3 * batch_size).to(device)
            #skeletons= skeleton.flatten().to(device)
            #_, targets = label.max(dim=1)
            #labels = targets
            #labels = labels.flatten().long().to(device)
            #labels = labels.reshape(-1, 1 * batch_size).long().to(device)
            #labels = label.flatten().to(device)
            #print(label)
                outputs = model(skeleton)            

                _, predicted = torch.max(outputs.data, 1)
                
                if not torch.eq(predicted, label):
                    count = count + 1
                print(count)           

               

    #model = FCL_no_activate(input_size, hidden_size, num_classes).to(device)



