'''
Image Classification on CIFAR-10
'''

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle
import numpy as np
import logging
import random

from resnet import *
from resnext import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model_arch', default="ResNet20", help='specify the model class you want to use')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay coefficient')
parser.add_argument('--test', action='store_true', help='resume from checkpoint')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--train_batch_size', default=128)
parser.add_argument('--test_batch_size', default=128)
parser.add_argument('--nepochs', default=160)
parser.add_argument('--seed', default=1234)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # correct the normalization by https://github.com/kuangliu/pytorch-cifar/issues/19
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model

# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
nepochs = args.nepochs
modelname = args.model_arch
checkpoint_savename = './checkpoint/{}.ckpt'.format(modelname)

net = eval(modelname)()

logf = open("log_160_{}".format(modelname), "a+")
# Training
def train(epoch):
    logf.write('\nEpoch: %d' % epoch)
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_errs = []
    batch_accs = []
    batch_losses = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loss = train_loss / (batch_idx + 1)
        acc = float(correct) / total
        logf.write('[%d]Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (batch_idx, loss, acc, correct, total))
        if batch_idx % 200 == 0:
            print('[%d]Loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, loss, acc, correct, total))
        batch_errs.append(1 - acc)
        batch_accs.append(acc)
        batch_losses.append(loss)
    return np.mean(batch_losses), np.mean(batch_errs), np.mean(batch_accs)

def test(epoch):
    global best_acc
    net.eval()
    test_loss, correct, total = 0, 0, 0
    batch_errs, batch_accs, batch_losses = [], [], []

    for batch_idx, (inputs, targets) in enumerate(testloader):

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loss = test_loss / (batch_idx + 1)
        acc = float(correct) / total
        logf.write('[%d] Val Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                   % (batch_idx, loss, acc, correct, total))
        batch_errs.append(1 - acc)
        batch_accs.append(acc)
        batch_losses.append(loss)

    acc = correct / total
    print('Val Acc:{} ({}/{})'.format(acc, correct, total))
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, checkpoint_savename)
        best_acc = acc

    return np.mean(batch_losses), np.mean(batch_errs), np.mean(batch_accs)


train_err, train_loss, train_acc = [], [], []
val_err, val_loss, val_acc = [], [], []
print('==> Building model..')
net = net.to(device)

criterion = nn.CrossEntropyLoss()

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_savename)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    trained_epoch = checkpoint['epoch']
    tl, te, ta = test(trained_epoch)
    print('trained_epoch:{} saved_acc:{} test loss:{} test_error:{} test_acc:{}'.format(
        trained_epoch, best_acc, tl, te, ta))

elif args.train:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    update_lr = {int(0.5 * nepochs): args.lr * 0.1, int(0.75 * nepochs): args.lr * 0.01}
    for epoch in range(0, nepochs):
        l, e, a = train(epoch)
        train_loss.append(l)
        train_err.append(e)
        train_acc.append(a)
        tl, te, ta = test(epoch)
        val_loss.append(tl)
        val_err.append(te)
        val_acc.append(ta)
        if epoch in update_lr:
            print("update learning rate to {}".format(update_lr[epoch]))
            optimizer = optim.SGD(net.parameters(), lr=update_lr[epoch], momentum=0.9, weight_decay=args.wd)
    result = {"train_err": train_err, "train_loss": train_loss, "train_acc": train_acc, \
              "val_loss": val_loss, "val_err": val_err, "val_acc": val_acc}

    fn = "./output/{}_start_epoch_{}_epochs_{}.pk".format(modelname, start_epoch, nepochs)
    if not os.path.exists('./output/'):
        os.mkdir('./output/')
    with open(fn, 'wb') as fout:
        pickle.dump(result, fout)
