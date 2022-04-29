import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np


class Medical_Dataset(data.Dataset):
    def __init__(self, root, trainsize=512,mode='train',augmentation_prob=0.4):
        self.trainsize = trainsize
        self.image_root = root
        self.gt_root = root[:-1]+'_GT/'
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png')
                    or f.endswith('.tif')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize,self.trainsize),Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), Image.NEAREST),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        #print(image.size)
        #image,gt = self.resize(image,gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        file_name = self.images[index].split('/')[-1][:-len(".tif")]
        return image, gt, file_name

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(root, batchsize, trainsize, num_workers=4, mode='train',augmentation_prob=0.4,shuffle=True, pin_memory=True):

    dataset = Medical_Dataset(root= root, trainsize= trainsize,mode =mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


