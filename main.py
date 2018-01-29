#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import pickle
import random
import datetime
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel

from model import *
from utils import *


label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
              'car', 'cat', 'chair', 'cow', 'dining', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']


def train(dataloader, epoch_num, class_num,
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
    start_epoch = get_start_epoch()
    net.train()
    for epoch in range(start_epoch, start_epoch+epoch_num):
        loss_sum = 0
        color_loss_sum = 0
        classify_loss_sum = 0
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
            color_loss_sum += colorization_loss
            classify_loss_sum += classify_loss
            print('batch: %d     ' %(i,), end='\r')
        print('epoch: %d  loss: %f  classify loss: %f  color loss: %f' %
              (epoch, loss_sum, classify_loss_sum, color_loss_sum))
        train_log(epoch, loss_sum, classify_loss_sum, color_loss_sum)
    print('Finished Training')


def test(dataloader, class_num, net, root):
    """
    """
    root = os.path.join(root, 'result')
    if not os.path.exists(root):
        os.makedirs(root)

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

        classify_result, result = net(gray_img)
        classify_result = torch.max(classify_result.data, 1)[1]

        total += label.size(0)
        correct += (classify_result == label).sum()

        if i%20 == 0:
            save_img(root, i, result.data,
                     gray_img.data, ab_img.data)
    print('Classification accuracy: %d %%' % (100 * correct / total))
    test_log(correct / total)


def main():
    data_root = 'data/'
    label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']
    trainloader, testloader = load_data(data_root, label_list)
    net = DataParallel(ColorizationNet(len(label_list)))

    model_path = 'colorization.pth'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    train(trainloader, 5, len(label_list), net=net, lr=0.01)
    torch.save(net.state_dict(), 'colorization.pth')
    save_log()
    # test(testloader, len(label_list), net, data_root)


if __name__ == '__main__':
    main()

