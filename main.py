#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import pickle
import random
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel

from model import *
from utils import *


label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'dining', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']


def train(dataloader, start_epoch, epoch_num, class_num,
          net=None, lr=0.001, alpha=0.0033):
    """
    """
    use_cuda = torch.cuda.is_available()
    if not net:
        net = DataParallel(ColorizationNet(class_num))
    classify_criterion = nn.CrossEntropyLoss()
    colorization_criterion = nn.MSELoss()
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    if use_cuda:
        net = net.cuda()
        classify_criterion = classify_criterion.cuda()
        colorization_criterion = colorization_criterion.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    print('====   Training..   ====')
    net.train()
    for epoch in range(start_epoch, start_epoch+epoch_num):
        loss_sum = 0
        for i, (gray_img, ab_img, label) in enumerate(dataloader):
            if use_cuda:
                gray_img = gray_img.cuda()
                ab_img = ab_img.cuda()
                label = label.cuda()
            gray_img = Variable(gray_img)
            ab_img = Variable(ab_img)
            label = Variable(label)

            optimizer.zero_grad()
            classify_result, result = net(gray_img)
            classify_loss = classify_criterion(classify_result, label)
            colorization_loss = colorization_criterion(result, ab_img)
            loss = colorization_loss + alpha * classify_loss
            loss.backward()
            optimizer.step()

            loss_sum += loss
            if i%10 == 0:
                print(loss)
        print('epoch: %d     loss: %f' % (epoch, loss_sum))
    print('Finished Training')
    return net


def test(dataloader, class_num, net):
    """
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    print('====    Testing..   ====')
    net.eval()

    total = correct = 0
    for i, (gray_img, ab_img, label) in enumerate(dataloader):
        if use_cuda:
            gray_img = gray_img.cuda()
            ab_img = ab_img.cuda()
            label = label.cuda()
        gray_img = Variable(gray_img)
        ab_img = Variable(ab_img)
        label = Variable(label)

        optimizer.zero_grad()
        classify_result, result = net(gray_img)
        classify_result = torch.max(classify_result.data, 1)[1]

        total += label.size(0)
        correct += (classify_result == label)
        print('Classification accuracy: %d %%' % (100 * correct / total))

        if i%10 == 0:
            show_img(result, gray_img, ab_img)


def main():
    label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']
    trainloader, testloader = load_data('data/', label_list)
    train(trainloader, 0, 1, len(label_list))


if __name__ == '__main__':
    main()

