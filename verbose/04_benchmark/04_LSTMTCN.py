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
from torch import nn
import argparse
import glob
from tensorboardX import SummaryWriter
import pandas as pd
from torchsummary import summary
from tcn import TemporalConvNet
from torch.nn.modules.module import _addindent

count = 0


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', type=str, default='finalLTSMTCN', help='task to be trained')
    parser.add_argument('-f', '--file', type=str, default='finalLTSMTCN', help='tensorboard location')
    parser.add_argument('-r', '--runs', type=str, default='final/LTSMTCN', help='tensorboard location')
    parser.add_argument('-b', '--batchsize', type=int, default=64, help='batchsize')
    parser.add_argument('-m', '--max', type=int, default=200, help='batchsize')
    parser.add_argument('-l', '--force_learning_rate', type=float, default=0.00001, help='setting learning rate')
    args = parser.parse_args()
    return args

opt = parse_command_line()
writer = SummaryWriter(opt.runs)
params = { 'batch_size': opt.batchsize, 'shuffle': True, 'num_workers': 10, 'drop_last': True}
learning_rate = opt.force_learning_rate

data = {'Data_hz': 2, 'Frame_len': 25}

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
                self.csv_result.append(tmp_list[-1])
        self.csv_conbined_df = np.concatenate(self.csv_list_of_dfs)
        self.csv_torch_tensor = torch.tensor(self.csv_conbined_df)
        #print(len(self.csv_result))
        print(self.csv_result[1])

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
        
        print("length of input skeleton is:"+str(len(self.npy_conbined_inputs))+" mod of batch size is:"+str(len(self.npy_conbined_inputs)%params['batch_size']))
        print("length of input label is:"+str(len(self.csv_conbined_df))+" mod of batch size is:"+str(len(self.csv_conbined_df)%params['batch_size']))

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

        #out = self.fc(out[:, -1, :])
        return out

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=7, dropout=0)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        #o = self.linear(y1)
        #return F.log_softmax(o, dim=1)
        #return o
        #o = self.linear(y1[:, :, -1])
        return o
        #return F.log_softmax(o, dim=1)



class LSTMTCN(nn.Module):
    def __init__(self, input_size1, output_size, num_channels,input_size2, hidden_size, num_layers, num_classes):
        super(LSTMTCN, self).__init__()
        self.lstm_block = simpleLSTM(input_size1, hidden_size, num_layers, num_classes)
        self.fcn_block = TCN(input_size2, output_size, num_channels,kernel_size=7,dropout=0)
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(num_channels[-1]+hidden_size, output_size)
    def forward(self, x):
        # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
        #x = x.unsqueeze(1)
        # pass input through LSTM block
        x1 = self.lstm_block(x)
        # pass input through FCN block
        x2 = self.fcn_block(x)
        # concatenate blocks output
        x = torch.cat([x1, x2], 1)
        print("!!!")
        print(x.size())
        # pass through Softmax activation
        #y = self.softmax(x)
        o = self.linear(x)
        return o

best_accu = 0

if __name__ == "__main__":
    with open(opt.file+'.txt', 'w') as f:

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        print(use_cuda)

        max_epochs = opt.max

        #csv_path = {'train':"../00_datasets/Julian_data/label_not5/S*.txt", 'val':"../00_datasets/Julian_data/label_5/S*.txt"}
        #npy_path = {'train':"../00_datasets/Weiling_data/pose_not5/S*.npy",'val':"../00_datasets/Weiling_data/pose_5/S*.npy"}
        csv_path = {'train':"../00_datasets/Weiling_data/label_not5/S*.csv", 'val':"../00_datasets/Weiling_data/label_5/S*.csv"}
        npy_path = {'train':"../00_datasets/Weiling_data/pose_not5/S*.npy",'val':"../00_datasets/Weiling_data/pose_5/S*.npy"}



        training_set = my_dataset(csv_path['train'], npy_path['train'], data['Data_hz'],data['Frame_len'])
        training_generator = DataLoader(training_set, **params)

        validation_set = my_dataset(csv_path['val'], npy_path['val'],data['Data_hz'],data['Frame_len'])
        validation_generator = DataLoader(validation_set, **params)

        need_print = True
        need_print_tail = True

        batch_size = params['batch_size']
        n_classes = 10
        input_channels = 3*32
        input_channels = 1
        seq_length = data['Frame_len']*3*32
        total_step = len(training_set)

        input_size = 32 * 3  
        hidden_size = 64
        num_layers = 1
        output_size = 10
        
        learning_rate = opt.force_learning_rate

        #model = simpleLSTM(input_size, hidden_size, num_layers, output_size)
        #model = TCN(input_channels, n_classes, [13] * 4, kernel_size=5, dropout=0.00)
        model = LSTMTCN(input_channels, n_classes, [25] * 8, input_size, hidden_size, num_layers, output_size)
        print(torch_summarize(model))


        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        sum_done = False
        for epoch in range(max_epochs):
            # Training
            model.train()
            for i, (skeleton, label) in enumerate(training_generator):

                skeleton = skeleton.reshape(-1,input_channels,seq_length).to(device)
                label = label.to(device)

                outputs = model(skeleton)

                #loss = F.nll_loss(outputs.view(-1, n_classes),label.view(-1))
                loss = criterion(outputs.view(-1, n_classes),label.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 1000 == 0:
                    count = count + 1000
                    writer.add_scalar('Train/Loss', loss.item(), global_step=count)
                    print('Train Epoch [{}/{}], Loss: {}'.format(str(epoch + 1), str(max_epochs), str(loss.item())),file=f)
                    print('Train Epoch [{}/{}], Loss: {}'.format(str(epoch + 1), str(max_epochs), str(loss.item())))

            model.eval()
            #Validation
            with torch.set_grad_enabled(False):
                correct = 0
                total = 0
                for skeleton_val, label_val in validation_generator:
                    skeleton_val = skeleton_val.reshape(-1,input_channels,seq_length).to(device)
                    label_val = label_val.to(device)
                
                    outputs = model(skeleton_val)
                    _, predicted = torch.max(outputs.data, 1)
                    #_, predicted = torch.max(outputs.data, 2)
                    total += label_val.size(0)
                    #total += label_val.size(0) * label_val.size(1)
                    correct += (predicted == label_val).sum().item()
                    
                    if need_print == True and epoch == 0:
                        print("Validation: head of each epoch")
                        print(skeleton_val.size())
                        print(label_val.size())
                        need_print = False
                if need_print_tail == True and epoch == 0:   
                    print("Validation: tail of each epoch")
                    print(skeleton_val.size())
                    print(label_val.size())
                    need_print_tail = False
                
                print('Test Epoch [{}/{}], Loss: {}'.format(str(epoch + 1), str(max_epochs), str(loss.item())),file=f)
                print('Test Accuracy: {}%'.format(100 * correct / total), file=f)
                if (100 * correct / total > best_accu):     
                    PATH = "../modelfinal/"+opt.file+"best.pth"
                    torch.save(model.state_dict(), PATH)
                    best_accu = 100 * correct / total
                writer.add_scalar('Test/Loss', loss.item(), global_step=epoch)
                writer.add_scalar('Test/Accuracy', 100 * correct / total, epoch)

                writer.flush()
        f.close()