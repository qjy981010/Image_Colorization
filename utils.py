import pickle
import torch
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageCms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class LogManager(object):
    """
    """
    def __init__(self, loss_label, log_file, log_iter, save_iter, total_iter):
        self.loss_label = loss_label
        self.log_file = log_file
        self.log_iter = log_iter
        self.save_iter = save_iter
        self.total_iter = total_iter

        self.iter_counter = self.get_start_iter()
        self.end_iter = self.iter_counter + total_iter
        self.start_time = datetime.datetime.now()
        self.loss = torch.zeros(len(loss_label))

        self.UPDATE = 0
        self.SAVE = 1
        self.EXIT = 2

    def get_start_iter(self):
        if not os.path.exists(self.log_file):
            os.mknod(self.log_file)
        with open(self.log_file, 'r+') as fp:
            lines = fp.readlines()
            fp.write('start\n')
            for i in range(1, len(lines)+1):
                if lines[-i][:5] == 'saved':
                    return int(lines[-i-1].split()[1])+1
        return 0

    def update(self, loss):
        self.loss += torch.Tensor(loss)
        self.iter_counter += 1

        if self.iter_counter%self.log_iter == 0 and self.iter_counter != 0:
            self.loss /= self.log_iter
            losses = [str(loss) for loss in self.loss]
            losses = (['iter %d'%(self.iter_counter)] +
                      [': '.join(x) for x in zip(self.loss_label, losses)] +
                      ['time: %s'%(datetime.datetime.now() - self.start_time)])
            loss_log = '  '.join(losses)

            print(loss_log)
            with open(self.log_file, 'a') as fp:
                fp.write(loss_log)
                fp.write('\n')

            self.loss.fill_(0)
            self.start_time = datetime.datetime.now()

            if self.iter_counter%self.save_iter == 0:
                with open(self.log_file, 'a') as fp:
                    fp.write('saved\n')
                return self.SAVE
        if self.iter_counter >= self.end_iter:
            return self.EXIT
        return self.UPDATE

    def test(self, accuracy):
        print('Classfication accuracy: %d %%' % (accuracy*100, ))
        with open(log_file, 'a') as fp:
            fp.write('Classfication accuracy: %d %%\n' % (accuracy*100, ))


class LabSaver(object):
    """
    """
    def __init__(self, root, save_root=None, name_ext='.pkl'):
        self.root = root
        if save_root == None:
            save_root = root
        self.save_root = save_root
        self.crop = transforms.Compose([
            transforms.Resize(256, Image.ANTIALIAS),
        ])

        srgb_profile = ImageCms.createProfile('sRGB')
        lab_profile = ImageCms.createProfile('LAB')
        self.labscale = ImageCms.buildTransformFromOpenProfiles(
            srgb_profile, lab_profile, 'RGB', 'LAB')

    def __call__(self, img_name):
        img = Image.open(os.path.join(self.root, img_name))
        img = self.crop(img)
        lab_img = ImageCms.applyTransform(img, self.labscale)
        img_name = img_name.split('.')[0]
        pickle.dump(lab_img, open(os.path.join(self.save_root, img_name+name_ext), 'wb'))


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
    for lab_img in lab_imgs[:10]:
        img = to_rgb(lab_img)
        img.save(os.path.join(root, '%d_%d.jpg'%(i, count)), 'JPEG')
        count += 1


def load_pascal(root, label_list):
    """
    """
    print('==== Loading Pascal VOC.. ====')
    lab_saver = LabSaver(os.path.join(root, 'JPEGImages'))
    img_list = get_img_list(os.path.join(root, 'ImageSets/Main'),
                            label_list, lab_saver)

    train_set = Pascal(img_list[:-10000], root)
    test_set = Pascal(img_list[-10000:], root)
    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


# for Pascal VOC dataset
def get_img_list(root, label_list, lab_saver):
    result = []
    pkl_file = os.path.join(root, 'img_list.pkl')
    if os.path.exists(pkl_file):
        result = pickle.load(open(pkl_file, 'rb'))
    else:
        label_encoder = {word: idx for idx, word in enumerate(label_list)}
        for file in os.listdir(root):
            filename = file[:-4].split('_')
            if len(filename) == 1 or filename[1] != 'trainval':
                continue
            label = label_encoder[filename[0]]
            with open(os.path.join(root, file), 'r') as fp:
                for line in fp.readlines():
                    img_name, posi = line.split()
                    result.append((label, img_name, posi == '1'))
                    lab_saver(img_name+'.jpg')
        pickle.dump(result, open(pkl_file, 'wb'))
    return result


class Pascal(Dataset):
    """
    """
    def __init__(self, img_list, root):
        super(Pascal, self).__init__()
        self.img_list = img_list
        self.root = os.path.join(root, 'JPEGImages')

        self.totensor = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        label, img_name, posi = self.img_list[idx]
        img = pickle.load(open(os.path.join(self.root, img_name+'pkl.jpg'), 'rb'))
        img = self.totensor(img)
        return img[0].unsqueeze(0), img[1:], label, posi


def folder2lab(root, save_root):
    if os.path.exists(save_root):
        return
    for folder in os.listdir(root):
        img_path = os.path.join(root, folder)
        save_path = os.path.join(save_root, folder)
        os.makedirs(save_path)
        lab_saver = LabSaver(img_path, save_path, name_ext='pkl.jpg')
        for img in os.listdir(img_path):
            lab_saver(img)


def pkl_loader(file):
    fp = open(file, 'rb')
    result = pickle.load(fp)
    fp.close()
    return result


class SunData(Dataset):
    """
    """
    def __init__(self, root, save_root):
        totensor = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        folder2lab(root, save_root)
        self.dataset = datasets.ImageFolder(root=save_root, transform=totensor,
                                            loader=pkl_loader)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            img, label = self.dataset[idx]
        except EOFError:
            return 
        return img[0].unsqueeze(0), img[1:], label, -1


def load_sun(root, save_root):
    """
    """
    print('==== Loading SUN dataset.. ====')
    dataset = SunData(root, save_root)
    loader = DataLoader(dataset, batch_size=28, shuffle=True, num_workers=4)
    return loader