import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, default='3d', help='task to be trained')

args = parser.parse_args()

if args.task == '3d':
    input_size = 96
elif args.task == 'euler':
    input_size = 96
elif ars.task == 'quat':
    input_size = 128
#hidden_size = 512
hidden_size = 512
num_classes = 7
#num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Fully connected layer with one hidden layer
class FCL(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = FCL(input_size, hidden_size, num_classes).to(device)

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