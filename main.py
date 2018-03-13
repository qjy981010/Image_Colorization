#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import os
import random
import datetime
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel

from model import *
from utils import *


def train(dataloader, iter_num, class_num, lr=1, alpha=0.0033,
          log_iter=100, save_iter=500, model_path='colorization.pth'):
    """
    """
    use_cuda = torch.cuda.is_available()

    net = ColorizationNet(class_num)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

    classify_criterion = nn.CrossEntropyLoss()
    colorization_criterion = nn.MSELoss()
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    if use_cuda:
        net = net.cuda()
        classify_criterion = classify_criterion.cuda()
        colorization_criterion = colorization_criterion.cuda()
    else:
        print("*****   Warning: Cuda isn't available!  *****")

    log_manager = LogManager(['class loss', 'color loss', 'total loss'],
                             'log.txt', log_iter, save_iter, iter_num)

    print('====   Training..   ====')
    net.train()
    while True:
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
            colorization_loss = colorization_criterion(result, ab_img)
            if posi[0] != -1:
                # for pascal
                posi = posi.nonzero().cuda()
                if posi.max() > 0:
                    classify_result = classify_result[posi, :].squeeze(1)
                    label = label[[int(x) for x in posi]]
                    classify_loss = classify_criterion(classify_result, label)
                else:
                    classify_loss = 0
            else:
                classify_loss = classify_criterion(classify_result, label)
            loss = colorization_loss + alpha * classify_loss
            loss.backward()
            optimizer.step()

            state = (log_manager.update([classify_loss.data[0], colorization_loss.data[0], loss.data[0]]))
            if state == log_manager.SAVE:
                print('saving...')
                torch.save(net.state_dict(), model_path)
                os.system('cp ' + model_path + ' ' + model_path[:-4] + '_copy.pth')
            elif state == log_manager.EXIT:
                return


def test(dataloader, class_num, model_path, root):
    """
    """
    root = os.path.join(root, 'result')
    if not os.path.exists(root):
        os.makedirs(root)

    net = ColorizationNet(class_num)
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))

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
        if posi[0] != -1:
            posi = posi.nonzero()
            if posi.max() > 0:
                classify_result = classify_result[posi, :].squeeze(1)
                label = label[posi, :].squeeze(1)
                classify_result = torch.max(classify_result.data, 1)[1]
                correct += (classify_result == label).sum()
                total += label.size(0)
        else:
            classify_result = torch.max(classify_result.data, 1)[1]
            correct += (classify_result == label).sum()
            total += label.size(0)

        save_img(root, i, result.data,
                 gray_img.data, ab_img.data)
    print('Classification accuracy: %d %%' % (100 * correct / total))
    test_log(correct / total)


def pascal_main():
    data_root = 'data/'
    pascal_label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                  'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                  'sheep', 'sofa', 'train', 'tvmonitor']
    trainloader, testloader = load_pascal(data_root, pascal_label_list)
    net = DataParallel(ColorizationNet(len(pascal_label_list)))

    model_path = 'colorization.pth'
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    train(trainloader, 100, len(pascal_label_list), net=net, lr=1, alpha=1/300)
    #save_log()
    #torch.save(net.state_dict(), 'colorization.pth')
    #test(testloader, len(pascal_label_list), net, data_root)


def sun_main():
    data_root = 'data/SUN'
    pkl_root = 'data/SUNpkl'
    loader = load_sun(data_root, pkl_root)
    class_num = 362
    net = ColorizationNet(class_num)
    model_path = 'output/colorization.pth'
    train(loader, 1000, class_num, lr=1, alpha=1/300, model_path=model_path)


if __name__ == '__main__':
    sun_main()

