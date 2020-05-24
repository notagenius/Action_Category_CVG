from __future__ import print_function, division
from finalsummy import torch_summarize
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from torch.nn.modules.module import _addindent
from torch import nn
import argparse
import glob
from tensorboardX import SummaryWriter
import pandas as pd
from torchsummary import summary
from finalsummy import torch_summarize
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import matplotlib.animation as animation
import sys


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', type=str, default='simpleLSTM', help='task to be trained')
    parser.add_argument('-f', '--file', type=str, default='simpleLSTM', help='tensorboard location')
    parser.add_argument('-r', '--runs', type=str, default='test/simpleLSTM', help='tensorboard location')
    parser.add_argument('-b', '--batchsize', type=int, default=64, help='batchsize')
    parser.add_argument('-m', '--max', type=int, default=200, help='batchsize')
    parser.add_argument('-l', '--force_learning_rate', type=float, default=0.00001, help='setting learning rate')
    args = parser.parse_args()
    return args

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

opt = parse_command_line()
writer = SummaryWriter(opt.runs)
params = { 'batch_size': opt.batchsize, 'shuffle': True, 'num_workers': 10, 'drop_last': True}
learning_rate = opt.force_learning_rate

data = {'Data_hz': 2, 'Frame_len': 100}

def default_loader(csv_path_folder, npy_path_folder):
    return

def get_filename_type(file):
    filename = file.split("/")[-1].split('.')[:-1]
    file_type = file.split("/")[-1].split('.')[-1]
    return filename, file_type


class my_dataset(Dataset):
    def __init__(self, csv_path_folder, npy_path_folder, data_hz, frame_len):
        
        self.data_hz = data_hz
        self.frame_len = frame_len

        # txt
        self.csv_filenames = sorted(glob.glob(csv_path_folder))
        self.csv_list_of_dfs = [np.loadtxt(filename, dtype=np.float32) for filename in self.csv_filenames]
        self.csv_dataframes = {}
        self.csv_filename = []
        self.csv_result = []
        for csv_dataframe, csv_filename in zip(self.csv_list_of_dfs, self.csv_filenames):
            tmp_name,_= get_filename_type(csv_filename)
            self.csv_filename.append(tmp_name)
            self.csv_dataframes[csv_filename] = csv_dataframe
        for i in self.csv_list_of_dfs:
            for j in range(len(i)-(self.frame_len-1)*self.data_hz):
                tmp_list=[]
                for k in range(self.frame_len):
                    tmp_list.append(i[j+k*self.data_hz].argmax(axis=0))
                self.csv_result.append(tmp_list)
        self.csv_conbined_df = np.concatenate(self.csv_list_of_dfs)
        self.csv_torch_tensor = torch.tensor(self.csv_conbined_df)


        # npy
        self.npy_filenames = sorted(glob.glob(npy_path_folder))
        self.npy_list_of_frames = [np.load(filename) for filename in self.npy_filenames]
        self.npy_inputs = {}
        self.npy_filename = []
        self.npy_result = []

        for i in self.npy_list_of_frames:
            for j in range(len(i)-(self.frame_len-1)*self.data_hz):
                tmp_list=[]
                for k in range(self.frame_len):
                    tmp_list.append(np.concatenate(i[j+k*self.data_hz]))
                self.npy_result.append(tmp_list)

        for npy_input, npy_filename in zip(self.npy_list_of_frames, self.npy_filenames):
            tmp_name,_= get_filename_type(npy_filename)
            if tmp_name not in self.csv_filename:
                self.npy_inputs[npy_filename] = npy_input
        self.npy_conbined_inputs = np.concatenate(self.npy_list_of_frames, axis=0, out=None)
        self.npy_torch_tensor = torch.tensor(self.npy_conbined_inputs)
        
    
    def __len__(self):
        return len(self.csv_result)

    def __getitem__(self, index):
        return np.asarray(self.npy_result[index]),np.asarray(self.csv_result[index])

class simpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(simpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (h_n, h_c) = self.lstm(x, (h0, c0))

        out = self.fc(out)
        #return F.log_softmax(out, dim=1)
        return out


if __name__ == "__main__":
    with open(opt.file+'.txt', 'w') as f:

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print(use_cuda)

        max_epochs = opt.max

        csv_path = {'train':"../00_datasets/Weiling_data/label_not5/S*.csv", 'val':"../00_datasets/Weiling_data/label_5/S*.csv"}
        npy_path = {'train':"../00_datasets/Weiling_data/pose_not5/S*.npy",'val':"../00_datasets/Weiling_data/pose_5/S*.npy"}

        training_set = my_dataset(csv_path['train'], npy_path['train'], data['Data_hz'],data['Frame_len'])
        training_generator = DataLoader(training_set, **params)

        validation_set = my_dataset(csv_path['val'], npy_path['val'],data['Data_hz'],data['Frame_len'])
        validation_generator = DataLoader(validation_set, **params)

        need_print = True
        need_print_tail = True

        batch_size = params['batch_size']
        time_step = data['Frame_len']
        input_size = 32 * 3  
        hidden_size = 64
        num_layers = 1
        output_size = 10
        total_step = len(training_set)
        
        learning_rate = opt.force_learning_rate

        model = simpleLSTM(input_size, hidden_size, num_layers, output_size)
        model.to(device)
        get_n_params(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        sum_done = False
        for epoch in range(max_epochs):
            model.train()
            for i, (skeleton, label) in enumerate(training_generator):

                skeleton = skeleton.reshape(-1, time_step, input_size).to(device)

                label = label.to(device)

                outputs = model(skeleton)

                loss = criterion(outputs.view(-1, output_size),label.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 1000 == 0:
                    writer.add_scalar('Train/Loss', loss.item(), global_step=epoch)
                    print('Train Epoch [{}/{}], Loss: {}'.format(str(epoch + 1), str(max_epochs), str(loss.item())),file=f)

            model.eval()
            with torch.set_grad_enabled(False):
                correct = 0
                total = 0
                for skeleton_val, label_val in validation_generator:
                    skeleton_val = skeleton_val.reshape(-1, time_step, input_size).to(device)
                    label_val = label_val.to(device)
                
                    outputs = model(skeleton_val)
                    _, predicted = torch.max(outputs.data, 2)
                    total += (label_val.size(0) * label_val.size(1))
                    correct += (predicted == label_val).sum().item()

                    loss = criterion(outputs.view(-1, output_size),label_val.view(-1))


                
                print('Test Epoch [{}/{}], Loss: {}'.format(str(epoch + 1), str(max_epochs), str(loss.item())),file=f)
                print('Test Accuracy: {}%'.format(100 * correct / total), file=f)
                PATH = "../model/"+opt.file+"best.pth"
                torch.save(model.state_dict(), PATH)
                writer.add_scalar('Test/Loss', loss.item(), global_step=epoch)
                writer.add_scalar('Test/Accuracy', 100 * correct / total, epoch)

                writer.flush()
        f.close()
    



