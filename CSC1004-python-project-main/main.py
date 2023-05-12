from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import multiprocessing
import matplotlib.pyplot as plt

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print('output = %s'%output)
        # print('loss = %s'%loss)
    '''Fill your code'''
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    total_correct += correct
    total_examples += target.size(0)
    total_loss += loss.item()

    training_acc = total_correct / total_examples
    training_loss = total_loss / len(train_loader)
    # print('training_acc = %s'%training_acc)
    # print('training_loss = %s' % training_loss)

    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            pass
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total_correct += correct
        total_examples += target.size(0)
        total_loss += loss.item()

        testing_acc = total_correct / total_examples
        testing_loss = total_loss / len(test_loader)
        # print('testing_acc = %s' % testing_acc)
        # print('testing_loss = %s' % testing_loss)

        return testing_acc, testing_loss


def plot(epoches, performance, filename):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    # print(len(epoches), len(performance))
    plt.plot(epoches, performance)
    plt.xlabel('Epoches')
    plt.ylabel('Performance')
    plt.title(filename)
    # plt.show()
    # print(epoches.shape)
    # print(epoches)
    # print(performance)
    plt.savefig(filename)

    pass
def clear():
    """
    Clear previous data.
    :return:
    """
    with open('test_acc_123.txt', 'r+') as test_acc1:
        test_acc1.truncate(0)
    with open('test_acc_321.txt', 'r+') as test_acc2:
        test_acc2.truncate(0)
    with open('test_acc_666.txt', 'r+') as test_acc3:
        test_acc3.truncate(0)
    with open('test_loss_123.txt', 'r+') as test_loss1:
        test_loss1.truncate(0)
    with open('test_loss_321.txt', 'r+') as test_loss2:
        test_loss2.truncate(0)
    with open('test_loss_666.txt', 'r+') as test_loss3:
        test_loss3.truncate(0)
    with open('train_acc_123.txt', 'r+') as train_acc1:
        train_acc1.truncate(0)
    with open('train_acc_321.txt', 'r+') as train_acc2:
        train_acc2.truncate(0)
    with open('train_acc_666.txt', 'r+') as train_acc3:
        train_acc3.truncate(0)
    with open('train_loss_123.txt', 'r+') as train_loss1:
        train_loss1.truncate(0)
    with open('train_loss_321.txt', 'r+') as train_loss2:
        train_loss2.truncate(0)
    with open('train_loss_666.txt', 'r+') as train_loss3:
        train_loss3.truncate(0)


def run(config, seed):
    # seeds = [123, 321, 666]
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    # for seed in seeds:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # torch.manual_seed(config.seed)
    # seeds = [123, 321, 666]
    # for seed in seeds:
    #     torch.manual_seed(seed)
    #     g = torch.Generator()
    #     g.manual_seed(seed)
    # g2 = torch.Generator()
    # g2.manual_seed(321)
    # g3 = torch.Generator()
    # g3.manual_seed(666)
    # g.manual_seed(config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 3,
                           'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

        # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs, generator=g)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = list(range(config.epochs))
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []
    tracc = []
    trloss = []
    teaccuracies = []
    teloss = []
    train_accs = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        tracc.append(train_acc)
        # train_accs['seed'].append(train_acc)
        trloss.append(train_loss)
        with open('train_acc_'+str(seed)+ '.txt', 'a') as f_acc, open('train_loss_'+str(seed)+'.txt', 'a') as f_loss:
            f_acc.write('%.4f' % train_acc + '\n')
            f_loss.write('%.4f' % train_loss + '\n')

        test_acc, test_loss = test(model, device, test_loader)
        """record testing info, Fill your code"""
        teaccuracies.append(test_acc)
        teloss.append(test_loss)
        with open('test_acc_'+str(seed)+'.txt', 'a') as t_acc, open('test_loss_'+str(seed)+'.txt', 'a') as t_loss:
            t_acc.write('%.4f' % test_acc + '\n')
            t_loss.write('%.4f' % test_loss + '\n')

        scheduler.step()
        """update the records, Fill your code"""

        """plotting training performance with the records"""

    plot(epoches, tracc, f"train_acc_{seed}.jpg")
    plt.clf()
    plot(epoches, trloss, f"train_loss_{seed}.jpg")
    plt.clf()

    #
    # """plotting testing performance with the records"""
    plot(epoches, teaccuracies, f"test_acc_{seed}.jpg")
    plt.clf()

    plot(epoches, teloss, f"test_loss_{seed}.jpg")
    plt.clf()

    #
    # if config.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean():
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    test_acc_123 = ReadTxtName('test_acc_123.txt')
    test_acc_321 = ReadTxtName('test_acc_321.txt')
    test_acc_666 = ReadTxtName('test_acc_666.txt')
    test_loss_123 = ReadTxtName('test_loss_123.txt')
    test_loss_321 = ReadTxtName('test_loss_321.txt')
    test_loss_666 = ReadTxtName('test_loss_666.txt')
    train_acc_123 = ReadTxtName('train_acc_123.txt')
    train_acc_321 = ReadTxtName('train_acc_321.txt')
    train_acc_666 = ReadTxtName('train_acc_666.txt')
    train_loss_123 = ReadTxtName('train_loss_123.txt')
    train_loss_321 = ReadTxtName('train_loss_321.txt')
    train_loss_666 = ReadTxtName('train_loss_666.txt')
    test_acc_mean = []
    test_loss_mean = []
    train_acc_mean = []
    train_loss_mean = []
    epoches = []
    for i in range(0, config.epochs):
        test_acc_mean.append(round(((float(test_acc_123[i]) + float(test_acc_321[i]) + float(test_acc_666[i])) / 3), 3))
        test_loss_mean.append(
            round(((float(test_loss_123[i]) + float(test_loss_321[i]) + float(test_loss_666[i])) / 3), 3))
        train_acc_mean.append(
            round(((float(train_acc_123[i]) + float(train_acc_321[i]) + float(train_acc_666[i])) / 3), 3))
        train_loss_mean.append(
            round(((float(train_loss_123[i]) + float(train_loss_321[i]) + float(train_loss_666[i])) / 3), 5))
        epoches.append(i + 1)
    """plotting mean training performance with the records"""
    plot(epoches, train_loss_mean, 'training_loss_mean.jpg')
    plt.clf()
    plot(epoches, train_acc_mean, 'training_accuracies_mean.jpg')
    plt.clf()
    """plotting mean testing performance with the records"""
    plot(epoches, test_acc_mean, 'testing_accuracies_mean.jpg')
    plt.clf()
    plot(epoches, test_loss_mean, 'testing_loss_mean.jpg')
    plt.clf()


def ReadTxtName(rootdir):
    """
    Function to read txt file and write the data into a list.
    :param rootdir:
    :return:
    """
    lines = []
    with open(rootdir, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
            # lines = int(lines)
    return lines
#     with open('train_acc_123.txt', 'r') as f:
#         lines = f.readlines()
#     train_acc_123 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         train_acc_123.append(last_data)
#
#     with open('train_acc_321.txt', 'r') as f:
#         lines = f.readlines()
#     train_acc_321 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         train_acc_321.append(last_data)
#
#     with open('train_acc_666.txt', 'r') as f:
#         lines = f.readlines()
#     train_acc_666 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         train_acc_666.append(last_data)
#     # --------------------------------------
#     with open('train_loss_123.txt', 'r') as f:
#         lines = f.readlines()
#     train_loss_321 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         train_loss_321.append(last_data)
#
#     with open('train_loss_321.txt', 'r') as f:
#         lines = f.readlines()
#     train_loss_321 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         train_loss_321.append(last_data)
#
#     with open('train_loss_666.txt', 'r') as f:
#         lines = f.readlines()
#     train_loss_666 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         train_loss_666.append(last_data)
# #     -----------------------------------------
#     with open('test_acc_123.txt', 'r') as f:
#         lines = f.readlines()
#     test_acc_123 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         test_acc_123.append(last_data)
#
#     with open('test_acc_321.txt', 'r') as f:
#         lines = f.readlines()
#     test_acc_321 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         test_acc_321.append(last_data)
#
#     with open('test_acc_666.txt', 'r') as f:
#         lines = f.readlines()
#     test_acc_666 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         test_acc_666.append(last_data)
# # --------------------------------------------------
#     with open('test_loss_123.txt', 'r') as f:
#         lines = f.readlines()
#     test_loss_123 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         test_loss_123.append(last_data)
#
#     with open('test_loss_321.txt', 'r') as f:
#         lines = f.readlines()
#     test_loss_321 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         test_loss_321.append(last_data)
#
#     with open('test_loss_666.txt', 'r') as f:
#         lines = f.readlines()
#     test_loss_666 = []
#     for line in lines:
#         data = line.strip().split(',')
#         last_data = float(data[:-1].strip(','))
#         test_loss_666.append(last_data)
#
#
#     train_acc_mean = []
#     for i in range(0, 2):
#         train_acc_mean[i] = (train_acc_123[i] + train_acc_321[i] + train_acc_666[i]) / 3
#     print(train_acc_mean)


if __name__ == '__main__':
    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    # run(config, seed)

    p1 = multiprocessing.Process(target=run, args=(config, 123))
    p2 = multiprocessing.Process(target=run, args=(config, 321))
    p3 = multiprocessing.Process(target=run, args=(config, 666))

    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()

    """plot the mean results"""
    plot_mean()
