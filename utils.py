import pickle
import torch
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageCms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class LabSaver(object):
    """
    """
    def __init__(self, root):
        self.root = root
        self.crop = transforms.Compose([
            transforms.Resize(256, Image.ANTIALIAS),
            transforms.RandomCrop(224),
        ])

        srgb_profile = ImageCms.createProfile('sRGB')
        lab_profile = ImageCms.createProfile('LAB')
        self.labscale = ImageCms.buildTransformFromOpenProfiles(
            srgb_profile, lab_profile, 'RGB', 'LAB')

    def __call__(self, img_name):
        img = Image.open(os.path.join(self.root, img_name+'.jpg'))
        img = self.crop(img)
        lab_img = ImageCms.applyTransform(img, self.labscale)
        pickle.dump(lab_img, open(os.path.join(self.root, img_name+'.pkl'), 'wb'),
                    pickle.HIGHEST_PROTOCOL)


def get_img_list(root, label_list, lab_saver, training):
    result = []
    train_val = 'train' if training else 'val'
    pkl_file = os.path.join(root, 'img_list_%s.pkl'%(train_val, ))
    if os.path.exists(pkl_file):
        result = pickle.load(open(pkl_file, 'rb'))
    else:
        label_encoder = {word: idx for idx, word in enumerate(label_list)}
        for file in os.listdir(root):
            filename = file[:-4].split('_')
            if len(filename) == 1 or filename[1] != train_val:
                continue
            with open(os.path.join(root, file), 'r') as fp:
                for line in fp.readlines():
                    img_name = line.split()[0]
                    result.append((label_encoder[filename[0]], img_name))
                    lab_saver(img_name)
        pickle.dump(result, open(pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    return result


class Pascal(Dataset):
    """
    """
    def __init__(self, root, label_list, training):
        super(Pascal, self).__init__()
        lab_saver = LabSaver(os.path.join(root, 'JPEGImages'))
        self.img_list = get_img_list(os.path.join(root, 'ImageSets/Main'),
                                     label_list, lab_saver, training)
        self.root = os.path.join(root, 'JPEGImages')

        self.totensor = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        label, img_name = self.img_list[idx]
        img = pickle.load(open(os.path.join(self.root, img_name+'.pkl'), 'rb'))
        img = self.totensor(img)
        return img[0].unsqueeze(0), img[1:], label


def load_data(root, label_list):
    """
    """
    print('==== Loading data.. ====')
    train_set = Pascal(root, label_list, True)
    test_set = Pascal(root, label_list, False)
    train_loader = DataLoader(train_set, batch_size=256,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=128,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


class LabTensorToRGB(object):
    """
    """
    def __init__(self):
        srgb_profile = ImageCms.createProfile('sRGB')
        lab_profile = ImageCms.createProfile('LAB')
        self.rgbscale = ImageCms.buildTransformFromOpenProfiles(
            lab_profile, srgb_profile, 'LAB', 'RGB'
        )

    def __call__(self, lab_tensor):
        lab_tensor = lab_tensor.mul(255).byte()
        np_img = np.transpose(lab_tensor.cpu().numpy(), (1, 2, 0))
        img = Image.fromarray(np_img, mode='LAB')
        img = ImageCms.applyTransform(img, self.rgbscale)
        return img


def save_img(root, i, out_ab_img, gray_img, ab_img):
    origin_img = torch.cat((gray_img, ab_img), 1)
    output_img = torch.cat((gray_img, out_ab_img), 1)
    lab_imgs = torch.cat((origin_img, output_img), 2)
    to_rgb = LabTensorToRGB()

    count = 0
    for lab_img in lab_imgs[:2]:
        img = to_rgb(lab_img)
        img.save(os.path.join(root, '%d_%d.jpg'%(i, count)), 'JPEG')


log_file = 'log.txt'

def train_log(epoch, loss):
    with open(log_file, 'a') as fp:
        fp.write('%s    epoch: %d    loss: %f\n' % (datetime.datetime.now(),
                                                    epoch, loss))


def get_start_epoch():
    if not os.path.exists(log_file):
        os.mknod(log_file)
    with open(log_file, 'a+') as fp:
        lines = fp.readlines()
        fp.write('start\n')
        for i in range(1, len(lines)+1):
            if lines[-i] == 'saved':
                return int(line[-i-1].split()[4])+1
    return 0


def save_log():
    with open(log_file, 'a') as fp:
        fp.write('saved\n')


def test_log(accuracy):
    with open(log_file, 'a') as fp:
        fp.write('Classfication accuracy: %d %%\n' % (accuracy*100, ))

