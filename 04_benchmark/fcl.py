import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils import data
from torch.utils.data.dataset import Dataset, DataLoader
from torchvision import transforms
import pandas as pd 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='3d', help='task to be trained')

args = parser.parse_args()

if args.task == '3d':
    input_size = 96
elif args.task == 'euler':
    input_size = 96
elif args.task == 'quat':
    input_size = 128
    
#hidden_size = 512
hidden_size = 32
num_classes = 10
num_epochs = 51455
batch_size = 64
learning_rate = 0.001

class MyDataset(Dataset):
    def __init__(self, record_path, is_train=True):
        self.data = []
        self.is_train = is_train
        with open(record_path) as fp:
            for line in fp.readlines():
                if line == '\n':
                    break
                else:
                    tmp = line.split("\t")
                    self.data.append(numpy.load([tmp[0]), pd.read_csv(tmp[1])])
        self.transformations = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
        img = self.transformations (Image.open(self.data[index][0]).resize((256,256)).convert('RGB'))
        label = int(self.data[index][1])
        return img, label
    def __len__(self):
        return len(self.data)

class Dataset(data.Dataset):
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y

model = FCL_no_activation(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#train
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (frame, labels) in enumerate(train_loader):
        #preparing the frame, ask Julian
        frames = frames.reshare(-1, 32 * 32).to(device)
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:,4f}'.format(epoch + 1, num_epochs, i+1, total_step, loss.item()))

#validation
with torch.no_grad():
    correct = 0
    total = 0
    for frames, labels in test_loader:
        frames = images.shape(-1).to(device)
        labels = labels.to(device)
        outputs = model(frames)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the S5 actor: {}%'.format(100 * correct / total))

torch.save(model.state_dict(), 'DNN-model.ckpt')