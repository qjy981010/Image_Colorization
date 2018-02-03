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

    ########
    #for p in net.module.mid_level_net.parameters():
    #    p.requires_grad = False
    #for p in net.module.fusion.parameters():
    #    p.requires_grad = False
    #for p in net.module.colorization.parameters():
    #    p.requires_grad = False
    ########
    classify_criterion = nn.CrossEntropyLoss()
    colorization_criterion = nn.MSELoss()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)# net.parameters(), lr=lr)
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
        batch_loss = 0
        start_time = datetime.datetime.now()
        for i, (gray_img, ab_img, label, posi) in enumerate(dataloader):
            if use_cuda:
                gray_img = gray_img.cuda()
                ab_img = ab_img.cuda()
                label = label.cuda()
            gray_img = Variable(gray_img)
            ab_img = Variable(ab_img)
            label = Variable(label)

            optimizer.zero_grad()
            classify_result, result = net(gray_img)
            ###################
            #x = net.module.low_level_net(gray_img)
            #classify_input = net.module.global_level_net(x)[0]
            #classify_result = net.module.classifier(classify_input)
            ###################
            posi = posi.nonzero().cuda()
            if posi.max() > 0:
                classify_result = classify_result[posi, :].squeeze(1)
                label = label[[int(x) for x in posi]]
                classify_loss = classify_criterion(classify_result, label)
            else:
                classify_loss = 0
            colorization_loss = colorization_criterion(result, ab_img)
            #colorization_loss = 0
            loss = colorization_loss + alpha * classify_loss
            loss.backward()
            optimizer.step()

            loss_sum += loss
            batch_loss += loss
            color_loss_sum += colorization_loss
            classify_loss_sum += classify_loss
            if i%100 == 0 and i != 0:
                print('batch: %d  loss: %f    ' % (i, batch_loss/100))
                batch_loss = 0
        end_time = datetime.datetime.now()
        print('epoch: %d  classify loss: %f  color loss: %f  loss: %f  time: %s' %
              (epoch, classify_loss_sum/(i+1), color_loss_sum/(i+1), loss_sum/(i+1), end_time-start_time))
        train_log(epoch, classify_loss_sum/(i+1), color_loss_sum/(i+1), loss_sum/(i+1))
        torch.save(net.state_dict(), 'colorization.pth')
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
    for i, (gray_img, ab_img, label, posi) in enumerate(dataloader):
        if use_cuda:
            gray_img = gray_img.cuda()
            ab_img = ab_img.cuda()
            label = label.cuda()
        gray_img = Variable(gray_img)
        ab_img = Variable(ab_img)

        classify_result, result = net(gray_img)
        posi = posi.nonzero()
        if posi.max() > 0:
            classify_result = classify_result[posi, :].squeeze(1)
            label = label[posi, :].squeeze(1)
            classify_result = torch.max(classify_result.data, 1)[1]
            correct += (classify_result == label).sum()
            total += label.size(0)
        else:
            classify_loss = 0

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
    train(trainloader, 200, len(label_list), net=net, lr=1, alpha=1/300)
    #save_log()
    #torch.save(net.state_dict(), 'colorization.pth')
    #test(testloader, len(label_list), net, data_root)


if __name__ == '__main__':
    main()

