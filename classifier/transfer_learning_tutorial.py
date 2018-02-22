# -*- coding: utf-8 -*-


from __future__ import print_function, division

import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime
import argparse
import sys
# import csv

plt.ion()   # interactive mode

parser = argparse.ArgumentParser(description = 'Hello World? :D')
parser.add_argument('--resume', '-r', action = 'store_true', help = 'resume from checkpoint')
parser.add_argument('filename', action='store', type = str)
parser.add_argument('--test', '-t', action='store_true', help = 'go for test mode')
args = parser.parse_args()

print('Hello! :D\n{}'.format(datetime.datetime.today()))



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/data/private/learn3/'
print('For %s'%data_dir)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) 
                  for x in ['train', 'val']}

print('Folder Loaded \t[1/3]')
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                             shuffle=True, num_workers=16)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


print('Dataset Loaded \t[2/3]')
use_gpu = torch.cuda.is_available()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

def train_model(model, criterion, optimizer, scheduler, num_epochs = 50, starting = 0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    record_idx = 0
    mmdd = datetime.datetime.today().strftime('%m%d')

    while os.path.isfile('./checkpoint/' + mmdd + str(record_idx) + '.t7'):
        record_idx += 1
    while os.path.isfile('./checkpoint/' + mmdd + str(record_idx) + '.csv'):
        record_idx += 1

    record_file = open('./checkpoint/' + mmdd + str(record_idx) + '.csv', 'w')

    for epoch in range(num_epochs):
        print('Epoch {}'.format(starting + epoch))
        print('-' * 10)
        idx = 0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            record_file.write('{}, {}, {} \n'.format(epoch, epoch_acc, epoch_loss) 
                + str(datetime.datetime.today()))
            idx += 1

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Saving..')
                state = {
                    'model': 'resnet50',
                    'net': model,
                    'acc': best_acc,
                    'epoch': epoch,
                    'time': datetime.datetime.now(),
                    'data': data_dir,
                    'class': os.listdir(data_dir + '/train'),
                }
                torch.save(state, './checkpoint/' + mmdd + str('_%03d'%record_idx) + '_temp.t7')
                torch.save(state, './checkpoint/' + mmdd + str('_%03d'%record_idx) + '.t7')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, criterion, optimizer, scheduler, num_epochs = 50, starting = 0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    record_idx = 0
    mmdd = datetime.datetime.today().strftime('%m%d')
    while os.path.isfile('./checkpoint/' + mmdd + str(record_idx) + '.t7'):
        record_idx += 1
    while os.path.isfile('./checkpoint/' + mmdd + str(record_idx) + '.csv'):
        record_idx += 1
    record_file = open('./checkpoint/' + mmdd + str(record_idx) + '.csv', 'w')
    print(record_idx)

    
    print('Epoch {}'.format(starting))
    print('-' * 10)
    idx = 0
    # Each epoch has a training and validation phase
    for phase in ['val']:
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if idx == 0:
                print(preds, labels.data)
            idx += 1
            writer.writerows([idx, labels.data, preds])

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        idx += 1


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model


# Set the number of classes 
num_class = sum(os.path.isdir(data_dir + '/train/' + i) for i in os.listdir(data_dir + '/train'))
cnn_model = ['resnet50', 'vgg19_bn']
idx = 0

if args.resume:
    # Load checkpoint
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    record_idx = 0
    if args.filename == '':
        mmdd = datetime.datetime.today().strftime('%m%d')
        while os.path.isfile('./checkpoint/' + mmdd + str('_%03d'%record_idx) + '.t7'):
            record_idx += 1

        checkpoint = torch.load('./checkpoint/' + mmdd + str('_%03d'%(record_idx - 1)) + '.t7')
    else: 
        checkpoint = torch.load('./checkpoint/' + args.filename)

    model_conv = checkpoint['net']
    start_epoch = checkpoint['epoch']

else:

    
    print("Model : %s" % cnn_model[idx])
    exec('model_conv = torchvision.models.' + cnn_model[idx] + '(pretrained=True)')

    for param in model_conv.parameters():
        param.requires_grad = False
    start_epoch = 0 
    


# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_class)

if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.1, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=50, gamma=0.5)

model_conv = nn.DataParallel(model_conv)
print('Model loaded \t[3/3]\n')

if args.test:
    test_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler)

else:
    print('{} \nLearning start'.format(datetime.datetime.today()))
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=300, starting = start_epoch)






