import pickle
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageCms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_img_list(root, label_list, training):
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
                    result.append((label_encoder[filename[0]],
                                   line.split()[0]))
            break
        pickle.dump(result, open(pkl_file, 'wb'), True)
    return result


class Pascal(Dataset):
    """
    """
    def __init__(self, root, label_list, training):
        super(Pascal, self).__init__()
        self.img_list = get_img_list(os.path.join(root, 'ImageSets/Main'),
                                     label_list, training)
        self.root = root
        self.crop = transforms.Compose([
            transforms.Resize(256, Image.ANTIALIAS),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
        self.grayscale = transforms.Grayscale()

        srgb_profile = ImageCms.createProfile('sRGB')
        lab_profile = ImageCms.createProfile('LAB')
        self.labscale = ImageCms.buildTransformFromOpenProfiles(
            srgb_profile, lab_profile, 'RGB', 'LAB')

        self.totensor = transforms.ToTensor()
        """
        self.data = []
        for label, img_name in self.img_list:
            img = Image.open(os.path.join(self.root, 'JPEGImages', img_name+'.jpg'))
            img = self.crop(img)
            gray_img = self.totensor(self.grayscale(img))
            lab_img = self.totensor(ImageCms.applyTransform(img, self.labscale))
            self.data.append((gray_img, lab_img[1:, :, :], label))
        """

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        label, img_name = self.img_list[idx]
        img = Image.open(os.path.join(self.root, 'JPEGImages', img_name+'.jpg'))
        img = self.crop(img)
        gray_img = self.totensor(self.grayscale(img))
        lab_img = self.totensor(ImageCms.applyTransform(img, self.labscale))
        return gray_img, lab_img[1:,:,:], label
        #return self.data[idx]


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
