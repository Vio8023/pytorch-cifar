'''
Image Classification on CIFAR-10
'''

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle
import numpy as np
import logging
import random

from resnet import *
from shakeshake_resnet import *
from resnext import *
from cutout import cutout
from collections import OrderedDict
from mixup import mixup_data, mixup_criterion
from noise import noise_data

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
parser.add_argument('--model_arch', default="ResNet20", help='specify the model class you want to use')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay coefficient')
parser.add_argument('--test', action='store_true', help='resume from checkpoint')
parser.add_argument('--train', action='store_true', help='train the model')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--nepochs', default=160)
parser.add_argument('--seed', default=1234)

################## These parameters are used in Cutout Model ####################
parser.add_argument('--use_cutout', action='store_true', default=False)
parser.add_argument('--use_post_cutout', action='store_true', default=False)
parser.add_argument('--cutout_size', type=int, default=16)
parser.add_argument('--cutout_prob', type=float, default=1)
parser.add_argument('--cutout_inside', action='store_true', default=False)
################## These parameters are used in Mix Up Model ####################
parser.add_argument('--use_mix_up',action="store_true", default=False)
parser.add_argument('--use_uniform_mixup',action="store_true", default=False)
parser.add_argument('--mix_up_alpha', type=float, default=0.2)
parser.add_argument('--prefix', type=str, default="exp")

################## These parameters are used in Noisy input ####################
parser.add_argument('--noise_type', type=str, default="gauss")
parser.add_argument('--noise_train', action='store_true', default=False)
parser.add_argument('--noise_test', action='store_true', default=False)

################## Concate
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = device is 'cuda'
best_acc, start_epoch = 0, 0

if args.noise_type is not None:
    if not args.noise_train:
        noise_func_train = noise_data(noise_type=args.noise_type, noise_prob=1.0)
    else:
        noise_func_train = noise_data(noise_type=args.noise_type, noise_prob=0.5)

    if not args.noise_test:
        noise_func_test = noise_data(noise_type=args.noise_type, noise_prob=1.0)
    else:
        noise_func_test = noise_data(noise_type=args.noise_type, noise_prob=0)
else:
    raise ValueError("unsupported noise type:{}".format(args.noise_type))

means = np.array([0.4914, 0.4822, 0.4465])
stds = np.array([0.2470, 0.2435, 0.2616])

if not args.use_cutout:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
        noise_func_train,
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        cutout(args.cutout_size,
               args.cutout_prob,
               args.cutout_inside),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        noise_func_train,
    ])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    noise_func_test,
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

trainset = trainset
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
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
    train_loss, correct, total = 0, 0, 0
    batch_accs = []
    batch_losses = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.use_mix_up:
            optimizer.zero_grad()
            inputs, targets_a, targets_b, lam = mixup_data(args, inputs, targets,
                                                           args.mix_up_alpha, args.use_uniform_mixup, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))

            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
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

        cur_loss = train_loss / (batch_idx + 1)
        acc = 100. * correct / total
        logf.write('[%d]Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (batch_idx, cur_loss, acc, correct, total))
        if batch_idx % 100 == 0:
            print('[%d]Loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx, cur_loss, acc, correct, total))
        batch_accs.append(acc)
        batch_losses.append(cur_loss)
    acc = float(correct) / total
    print('Train Acc:{}'.format(acc))
    return np.mean(batch_losses), acc


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

    acc = float(correct) / total
    print('Val Acc:{} ({}/{})'.format(acc, correct, total))
    return np.mean(batch_losses), acc


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
    print('==> Resuming from checkpoint {}..'.format(checkpoint_savename))
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_savename)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    trained_epoch = checkpoint['epoch']
    tl, ta = test(trained_epoch)
    print('trained_epoch:{} saved_acc:{} test loss:{} test_acc:{}'.format(
        trained_epoch, best_acc, tl, ta))

elif args.train:

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    update_lr = {int(0.5 * nepochs): args.lr * 0.1, int(0.75 * nepochs): args.lr * 0.01}
    for epoch in range(0, nepochs):
        l, a = train(epoch)
        train_loss.append(l)
        train_acc.append(a)
        tl, ta = test(epoch)
        val_loss.append(tl)
        val_acc.append(ta)
        if epoch in update_lr:
            print("update learning rate to {}".format(update_lr[epoch]))
            optimizer = optim.SGD(net.parameters(), lr=update_lr[epoch], momentum=0.9, weight_decay=args.wd)
    acc = ta
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, checkpoint_savename)
    best_acc = acc

    result = {"train_loss": train_loss, "train_acc": train_acc, \
              "val_loss": val_loss, "val_acc": val_acc}

    fn = "./output/{}_{}_start_epoch_{}_epochs_{}_noise{}.pk".format(
        args.prefix, modelname, start_epoch, nepochs, args.noise_type)
    if not os.path.exists('./output/'):
        os.mkdir('./output/')
    with open(fn, 'wb') as fout:
        pickle.dump(result, fout)
