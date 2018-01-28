import pickle
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageCms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class LabTransformer(object):
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
        img = Image.open(os.path.join(self.root, 'JPEGImages', img_name+'.jpg'))
        img = self.crop(img)
        lab_img = ImageCms.applyTransform(img, self.labscale)
        pickle.dump(lab_img, os.join(self.root, img_name+'.pkl'),
                    pickle.HIGHEST_PROTOCOL)


def get_img_list(root, label_list, lab_transformer, training):
    result = []
    train_val = 'train' if training else 'val'
    pkl_file = os.path.join(root, 'img_list_%s.pkl'%(train_val, ))
    if os.path.exists(pkl_file):
        result = pickle.load(open(pkl_file, 'rb'))
    else:
        os.system('rm '+os.path.join(root, '*.pkl'))
        label_encoder = {word: idx for idx, word in enumerate(label_list)}
        for file in os.listdir(root):
            filename = file[:-4].split('_')
            if len(filename) == 1 or filename[1] != train_val:
                continue
            with open(os.path.join(root, file), 'r') as fp:
                for line in fp.readlines():
                    img_name = line.split()[0]
                    result.append((label_encoder[filename[0]], img_name))
                    lab_transformer(img_name)
        pickle.dump(result, open(pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    return result


class Pascal(Dataset):
    """
    """
    def __init__(self, root, label_list, training):
        super(Pascal, self).__init__()
        lab_transformer = LabTransformer(os.path.join(root, 'JPEGImages'))
        self.img_list = get_img_list(os.path.join(root, 'ImageSets/Main'),
                                     label_list, lab_transformer, training)
        self.root = os.path.join(root, 'JPEGImages')

        self.totensor = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        label, img_name = self.img_list[idx]
        img = pickle.load(open(os.path.join(self.root, img_name+'.pkl')), 'rb')
        img = self.totensor(img)
        return img[0, :, :], img[1:,:,:], label


def load_data(root, label_list):
    """
    """
    print('==== Loading data.. ====')
    train_set = Pascal(root, label_list, True)
    test_set = Pascal(root, label_list, False)
    train_loader = DataLoader(train_set, batch_size=256,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


def show_img(out_ab_img, gray_img, ab_img):
    origin_img = torch.cat((gray_img, ab_img), 1)
    output_img = torch.cat((gray_img, out_ab_img), 1)
    unloader = transforms.ToPILImage()

    origin_img = origin_img.data.cpu()
    origin_img = unloader(origin_img)
    output_img = output_img.data.cpu()
    output_img = unloader(output_img)

    f, axarr = plt.subplots(1, 2)
    axarr[0, 0].set_title('origin')
    axarr[0, 0].imshow(origin_img)
    axarr[0, 1].set_title('output')
    axarr[0, 1].imshow(output_img)
