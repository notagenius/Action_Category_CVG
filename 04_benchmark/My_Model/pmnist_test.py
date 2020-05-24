import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from utils import data_generator
from model import TCN
import numpy as np
import argparse
from tensorboardX import SummaryWriter
from torchsummary import summary
from torch.nn.modules.module import _addindent

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.00,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-5,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')

parser.add_argument('-n', '--net', type=str, default='TCN_ori', help='task to be trained')
parser.add_argument('-f', '--file', type=str, default='TCN_ori', help='tensorboard location')
parser.add_argument('-r', '--runs', type=str, default='TCN_ori', help='tensorboard location')
parser.add_argument('-b', '--batchsize', type=int, default=64, help='batchsize')
parser.add_argument('-m', '--max', type=int, default=200, help='batchsize')
parser.add_argument('-l', '--force_learning_rate', type=float, default=0.00001, help='setting learning rate')
    
args = parser.parse_args()

opt = args
writer = SummaryWriter(args.runs)

number = 25*3*32
with open(args.file+'.txt', 'w') as f:

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    root = './data/mnist'
    batch_size = args.batch_size
    n_classes = 10
    input_channels = 1
    #seq_length = int(784 / input_channels)
    seq_length = int(number / input_channels)
    epochs = args.epochs
    steps = 0

    print(args)
    train_loader, test_loader = data_generator(root, batch_size)

    permute = torch.Tensor(np.random.permutation(number).astype(np.float64)).long()
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize
    model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
    print(torch_summarize(model))

    if args.cuda:
        model.cuda()
        permute = permute.cuda()

    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    def train(ep):
        global steps
        train_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda: data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_loss += loss
            steps += seq_length
            
            if batch_idx > 0 and batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                    ep, batch_idx * batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
                train_loss = 0
                writer.add_scalar('Train/Loss', loss.item(), global_step=steps)
                print('Train Epoch [{}/{}], Loss: {}'.format(str(epoch + 1), str("dumm"), str(loss.item())),file=f)




    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data = data.view(-1, input_channels, seq_length)
                if args.permute:
                    data = data[:, :, permute]
                data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_loader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            #print('Test Epoch [{}/{}], Loss: {}'.format(str(epoch + 1), str("dummy"), str(loss.item())),file=f)
            #print('Test Accuracy: {}%'.format(100 * correct / total), file=f)
            PATH = "./model/"+args.file+"best.pth"
            torch.save(model.state_dict(), PATH)
            writer.add_scalar('Test/Loss', test_loss, global_step=epoch)
            writer.add_scalar('Test/Accuracy', correct, global_step=epoch)

            writer.flush()
            return test_loss


    if __name__ == "__main__":
        for epoch in range(1, epochs+1):
            train(epoch)
            test()
            if epoch % 10 == 0:
                lr /= 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
