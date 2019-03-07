'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
sys.path.insert(0, "./")
sys.path.insert(0, "./models")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle

from models import *
# from utils import progress_bar

default_lr = 0.1
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=default_lr, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

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
# for modelname, net in zip(["ResNeXt29_2x64d"], [ResNeXt29_2x64d()]):
# for modelname, net in zip(["ResNet18", "ResNeXt29_2x64d"], [ResNet18(), ResNeXt29_2x64d()]):
for modelname, net in zip(["ResNet20"], [ResNet20()]):
    logf = open("log_160_{}".format(modelname), "a+")
    # Training
    def train(epoch):
        logf.write('\nEpoch: %d' % epoch)
        print('\nEpoch: %d' % epoch)
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
            acc = 100. * correct / total
            logf.write('[%d]Loss: %.3f | Acc: %.3f%% (%d/%d)\n'% (batch_idx, loss, acc, correct, total))
            print('[%d]Loss: %.3f | Acc: %.3f%% (%d/%d)'% (batch_idx, loss, acc, correct, total))
            batch_errs.append(1 - acc)
            batch_accs.append(acc)
            batch_losses.append(loss)
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return np.mean(batch_losses), np.mean(batch_errs), np.mean(batch_accs)

    import numpy as np
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        batch_errs = []
        batch_accs = []
        batch_losses = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()


                loss = test_loss/(batch_idx+1)
                acc = 100.*correct/total
                logf.write('[%d] Val Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                    % (batch_idx, loss, acc, correct, total))
                print('[%d] Val Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx, loss, acc, correct, total))
                batch_errs.append(1-acc)
                batch_accs.append(acc)
                batch_losses.append(loss)
                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_160.t7')
            best_acc = acc

        return np.mean(batch_losses), np.mean(batch_errs), np.mean(batch_accs)

    nepochs = 1
    train_err = []
    train_loss = []
    train_acc = []
    val_err = []
    val_loss = []
    val_acc = []
    print('==> Building model..')
    # net = ShuffleNetV2(1)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt_160.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    update_lr = {0.5*nepochs: default_lr*0.1, 0.75*nepochs: default_lr*0.01}

    for epoch in range(start_epoch, start_epoch+nepochs):
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
            optimizer = optim.SGD(net.parameters(), lr=update_lr[epoch], momentum=0.9, weight_decay=1e-4)

    result = {"train_err": train_err, "train_loss": train_loss, "train_acc": train_acc,\
              "val_loss": val_loss, "val_err": val_err, "val_acc": val_acc}
    fn = "{}_start_epoch_{}_epochs_{}".format(modelname, start_epoch, nepochs)
    fo = open(fn, "wb")
    pickle.dump(result, fo)
    fo.close()

    try:
        PATH = fn+"_save_state_160"
        torch.save(net.state_dict(), PATH)
    except Exception as e:
        print("save state failed:", e)

