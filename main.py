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
from resnet import *
from resnext import *
from cutout import cutout
from collections import OrderedDict
from mixup import mixup_data



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-4, help='weight decay coefficient')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--train_batch_size', default=128)
parser.add_argument('--test_batch_size', default=128)
parser.add_argument('--use_cutout', action='store_true', default=False)
parser.add_argument('--cutout_size', type=int, default=16)
parser.add_argument('--cutout_prob', type=float, default=1)
parser.add_argument('--cutout_inside', action='store_true', default=False)
parser.add_argument('--use_mix_up',action="store_true", default=False)
parser.add_argument('--mix_up_alpha', type=float, default=1)

args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = device is 'cuda'
best_acc, start_epoch = 0, 0

if not args.use_cutout:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # correct the normalization by https://github.com/kuangliu/pytorch-cifar/issues/19
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cutout(args.cutout_size,
               args.cutout_prob,
               args.cutout_inside),
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
nepochs = 160
checkpoint_savename = './checkpoint/ckpt_resnet_paper_{}.t7'.format(nepochs)

for modelname, net in zip(["ResNet20"], [ResNet20()]):
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
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           args.alpha, use_cuda)
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
            acc = 100. * correct / total
            logf.write('[%d]Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (batch_idx, loss, acc, correct, total))
            if batch_idx % 20 == 0:
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

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                loss = test_loss / (batch_idx + 1)
                acc = 100. * correct / total
                logf.write('[%d] Val Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                           % (batch_idx, loss, acc, correct, total))
                if batch_idx % 20 == 0:
                    print('[%d] Val Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (batch_idx, loss, acc, correct, total))
                batch_errs.append(1 - acc)
                batch_accs.append(acc)
                batch_losses.append(loss)

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
            torch.save(state, checkpoint_savename + str(epoch))
            best_acc = acc

        return np.mean(batch_losses), np.mean(batch_errs), np.mean(batch_accs)


    train_err = []
    train_loss = []
    train_acc = []
    val_err = []
    val_loss = []
    val_acc = []
    print('==> Building model..')
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(checkpoint_savename)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    update_lr = {int(0.5 * nepochs): args.lr * 0.1, int(0.75 * nepochs): args.lr * 0.01}

    for epoch in range(start_epoch, start_epoch + nepochs):
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
    fn = "/output/{}_start_epoch_{}_epochs_{}".format(modelname, start_epoch, nepochs)

    with open(fn, 'wb') as fout:
        pickle.dump(result, fout)

    torch.save(net.state_dict(), checkpoint_savename)
