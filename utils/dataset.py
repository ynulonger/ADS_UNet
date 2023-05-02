import cv2
import glob
import torch
import random
import logging
import numpy as np
import os.path as osp
from PIL import Image
from os import listdir
from os.path import splitext
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

def DataAug(img, target):
    i, j, h, w = transforms.RandomCrop.get_params(img, (512,512))
    img = F.crop(img, i, j, h, w)
    target = F.crop(target, i, j, h, w)
    if random.random() > 0.5:
        img = F.hflip(img)
        target = F.hflip(target)
    if random.random() > 0.5:
        img = F.vflip(img)
        target = F.vflip(target)
    return img, target

def KumarDataAug(img, target):
    if random.random() > 0.5:
        img = F.hflip(img)
        target = F.hflip(target)
    if random.random() > 0.5:
        img = F.vflip(img)
        target = F.vflip(target)
    return img, target

class Digest(Dataset):
    def __init__(self, img_pth, mask_pth, sample_weight=None, mask_channel=1):   # initial logic happens like transform
        self.image_paths = img_pth
        self.mask_paths = mask_pth
        self.mask_channel = mask_channel
        self.sample_weight =sample_weight
        if self.sample_weight == 'mean':
            self.sample_weight = torch.ones(len(self.image_paths),1,512,512)/(512**2*len(self.image_paths))
        elif self.sample_weight=='sample_mean':
            self.sample_weight = torch.ones(len(self.image_paths),1)/len(self.image_paths)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.6633, 0.3966, 0.6155), (0.1400, 0.2150, 0.1947))
        ])
    def __getitem__(self, index):
        # print('image:',self.image_paths[index],'mask:',self.mask_paths[index])
        assert(self.image_paths[index].split('/')[-1]==self.mask_paths[index].split('/')[-1]), 'img name is not the same as the mask name'
        image = cv2.imread((self.image_paths[index]),flags=1)
        # image = cv2.imread((self.image_paths[index]))
        mask  = cv2.imread(self.mask_paths[index],flags=0)//255
        mask = np.expand_dims(mask,axis=0)

        if self.sample_weight==None:
            return {
                'image': self.transforms(image),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'name': self.image_paths[index].split('/')[-1],
            }
        else:
            return {
                # 'image': torch.from_numpy(image).type(torch.FloatTensor),
                'image': self.transforms(image),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'name': self.image_paths[index].split('/')[-1],
                'weight': self.sample_weight[index]
            }
    def __len__(self):  # return count of sample we have
        return len(self.image_paths)

def my_transforms(img, mask, patch_size):
    height = img.shape[0]
    width = img.shape[1]
    top  = np.random.randint(1, height-patch_size)
    left = np.random.randint(1, width-patch_size)
    img_croped = img[top:top+patch_size, left: left+patch_size,:]
    mask_croped = mask[top:top+patch_size, left:left+patch_size]
    return img_croped, mask_croped

def mapping(gt, channels):
    size = gt.shape
    mask = np.zeros([channels, size[0], size[1]])
    labels = np.unique(gt)
    # print('labels:', labels, mask.shape)
    for label in labels:
        mask[label, :,:] = (gt==label).astype(np.uint8)
    return mask

class Kumar(Dataset):
    def __init__(self, image_set='train', sample_weight=None, mask_channel=2, fold=0):
        self.sample_weight=sample_weight
        self._image_set = image_set
        self.mask_channel = mask_channel

        dataset = np.load("/scratch/yy3u19/DataSets/dataset.npy", allow_pickle=True).item()
        num_imgs = len(dataset['Kumar_img'])
        num_per_fold = num_imgs//5
        idx = [i for i in range(num_imgs)]
        test_idx = [i for i in range(fold*num_per_fold, (fold+1)*num_per_fold)] if fold<4 else [i for i in range(fold*num_per_fold, num_imgs)]
        train_idx= [item for item in idx if item not in test_idx]
        if self._image_set == 'train':
            self._image_names= np.array(dataset['Kumar_img'])[train_idx]       
            self._mask_names = np.array(dataset['Kumar_mask'])[train_idx]
            # self._image_names = glob.glob('/scratch/yy3u19/DataSets/CRAG/TRAIN/img/*png')
            # self._mask_names = glob.glob('/scratch/yy3u19/DataSets/CRAG/TRAIN/labelcol/*png')

        elif self._image_set == 'test':
            self._image_names= np.array(dataset['Kumar_img'])[test_idx]       
            self._mask_names = np.array(dataset['Kumar_mask'])[test_idx]

        else:
            raise RuntimeError('image set should only be train or set')

        self.MEAN = (0.6519, 0.5859, 0.7658)
        self.STD = (0.1911, 0.2308, 0.2018)

        self.transforms  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])
        if self.sample_weight == 'mean':
            self.sample_weight = torch.ones(len(self._image_names),1,400,400)/(400**2*len(self._image_names))
        elif self.sample_weight=='sample_mean':
            self.sample_weight = torch.ones(len(self._image_names),1)/len(self._image_names)

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        image = cv2.imread('/scratch/yy3u19/DataSets/'+self._image_names[index], flags=1)
        mask  = cv2.imread('/scratch/yy3u19/DataSets/'+self._mask_names[index], flags=0)//255
        # mask  = np.expand_dims(mask,axis=0)
        mask  = mapping(mask, self.mask_channel)

        img  = self.transforms(image)
        mask = torch.from_numpy(mask).type(torch.float32)
        if self._image_set=='train':
            img, mask = KumarDataAug(img, mask)
        # elif self._image_set=='test':
        #     img = F.center_crop(img, (992,992))
        #     mask = F.center_crop(mask, (992,992))

        if self.sample_weight==None:
            return {
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
            }
        else:
            return {
                # 'image': torch.from_numpy(image).type(torch.FloatTensor),
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
                'weight': self.sample_weight[index]
            }

class GlaS(Dataset):
    def __init__(self, image_set='train', sample_weight=None):
        self.sample_weight=sample_weight
        self._image_set = image_set
        if self._image_set == 'train':
            self._image_names = glob.glob('/scratch/yy3u19/DataSets/GlaS/Train_patch/*')
            self._mask_names = glob.glob('/scratch/yy3u19/DataSets/GlaS/Train_patch_mask/*')

        elif self._image_set == 'val':
            # self._image_names = [img for img in glob.iglob(image_fp) if os.path.basename(img) in valids]
            self._image_names= glob.glob('/scratch/yy3u19/DataSets/GlaS/Test_patch/*png')
            self._mask_names= glob.glob('/scratch/yy3u19/DataSets/GlaS/Test_patch_mask/*png')
        elif self._image_set == 'test':
            # self._image_names = [img for img in glob.iglob(image_fp) if os.path.basename(img) in valids]
            self._image_names= glob.glob('/scratch/yy3u19/DataSets/GlaS/test_img/*png')
            self._mask_names= glob.glob('/scratch/yy3u19/DataSets/GlaS/test_gt/*png')
        else:
            raise RuntimeError('image set should only be train or set')
        self._image_names.sort()
        self._mask_names.sort()
        self.MEAN = (0.6535, 0.5831, 0.7515)
        self.STD = (0.1994, 0.2388, 0.2010)

        self.transforms  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])
        if self.sample_weight == 'mean':
            self.sample_weight = torch.ones(len(self._image_names),1,400,400)/(400**2*len(self._image_names))
        elif self.sample_weight=='sample_mean':
            self.sample_weight = torch.ones(len(self._image_names),1)/len(self._image_names)

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        assert(self._image_names[index].split('/')[-1]==self._mask_names[index].split('/')[-1]), self._image_names[index].split('/')[-1]+'--'+self._mask_names[index].split('/')[-1]

        image = cv2.imread(self._image_names[index], flags=1)
        mask  = cv2.imread(self._mask_names[index], flags=0)!=0
        mask  = np.expand_dims(mask,axis=0)

        img  = self.transforms(image)
        mask = torch.from_numpy(mask).type(torch.float32)

        if self.sample_weight==None:
            return {
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
            }
        else:
            return {
                # 'image': torch.from_numpy(image).type(torch.FloatTensor),
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
                'weight': self.sample_weight[index]
            }

class BCSS(Dataset):
    def __init__(self, image_set='train', sample_weight=None, mask_channel=5,fold=0):
        self.sample_weight=sample_weight
        self._image_set = image_set
        self.mask_channel = mask_channel


        dataset = np.load("/scratch/yy3u19/DataSets/dataset.npy", allow_pickle=True).item()
        num_imgs = len(dataset['BCSS_img'])
        num_per_fold = num_imgs//5
        idx = [i for i in range(num_imgs)]
        test_idx = [i for i in range(fold*num_per_fold, (fold+1)*num_per_fold)] if fold<4 else [i for i in range(fold*num_per_fold, num_imgs)]
        train_idx= [item for item in idx if item not in test_idx]
        if self._image_set == 'train':
            self._image_names= np.array(dataset['BCSS_img'])[train_idx]       
            self._mask_names = np.array(dataset['BCSS_mask'])[train_idx]
            # self._image_names = glob.glob('/scratch/yy3u19/DataSets/CRAG/TRAIN/img/*png')
            # self._mask_names = glob.glob('/scratch/yy3u19/DataSets/CRAG/TRAIN/labelcol/*png')

        elif self._image_set == 'test':
            self._image_names= np.array(dataset['BCSS_img'])[test_idx]       
            self._mask_names = np.array(dataset['BCSS_mask'])[test_idx]


        elif self._image_set == 'evaluate':
            # self._image_names = [img for img in glob.iglob(image_fp) if os.path.basename(img) in valids]
            self._image_names= glob.glob('/scratch/yy3u19/DataSets/BCSS/new_imgs/test/*png')
            self._mask_names= glob.glob('/scratch/yy3u19/DataSets/BCSS/new_gts/test/*png')
        else:
            raise RuntimeError('image set should only be train or set')
        # self._image_names.sort()
        # self._mask_names.sort()
        self.MEAN = (0.7258, 0.6042, 0.8183)
        self.STD  = (0.1708, 0.2302, 0.1744)

        self.transforms  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])
        if self.sample_weight=='sample_mean':
            self.sample_weight = torch.ones(len(self._image_names),1)/len(self._image_names)

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        assert(self._image_names[index].split('/')[-1]==self._mask_names[index].split('/')[-1]), self._image_names[index].split('/')[-1]+'--'+self._mask_names[index].split('/')[-1]

        image = cv2.imread('/scratch/yy3u19/DataSets/'+self._image_names[index], flags=1)
        mask  = cv2.imread('/scratch/yy3u19/DataSets/'+self._mask_names[index], flags=0)
        mask  = mapping(mask, self.mask_channel)

        img  = self.transforms(image)
        mask = torch.from_numpy(mask).type(torch.float32)

        if self._image_set == 'train':
            img, mask = DataAug(img, mask)
        elif  self._image_set == 'test':
            img = F.center_crop(img, (512,512))
            mask = F.center_crop(mask, (512,512))
            # print('img:',img.shape,'mask:',mask.shape)

        if self.sample_weight==None:
            return {
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
            }
        else:
            return {
                # 'image': torch.from_numpy(image).type(torch.FloatTensor),
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
                'weight': self.sample_weight[index]
            }

class CRAG(Dataset):
    def __init__(self, image_set='train', sample_weight=None, mask_channel=2, fold=0):
        self.sample_weight=sample_weight
        self._image_set = image_set
        self.mask_channel = mask_channel
        
        dataset = np.load("/scratch/yy3u19/DataSets/dataset.npy", allow_pickle=True).item()
        num_imgs = len(dataset['CRAG_img'])
        num_per_fold = num_imgs//5
        idx = [i for i in range(num_imgs)]
        test_idx = [i for i in range(fold*num_per_fold, (fold+1)*num_per_fold)] if fold<4 else [i for i in range(fold*num_per_fold, num_imgs)]
        train_idx= [item for item in idx if item not in test_idx]
        if self._image_set == 'train':
            self._image_names= np.array(dataset['CRAG_img'])[train_idx]       
            self._mask_names = np.array(dataset['CRAG_mask'])[train_idx]
            # self._image_names = glob.glob('/scratch/yy3u19/DataSets/CRAG/TRAIN/img/*png')
            # self._mask_names = glob.glob('/scratch/yy3u19/DataSets/CRAG/TRAIN/labelcol/*png')

        elif self._image_set == 'test':
            self._image_names= np.array(dataset['CRAG_img'])[test_idx]       
            self._mask_names = np.array(dataset['CRAG_mask'])[test_idx]
            # self._image_names= glob.glob('/scratch/yy3u19/DataSets/CRAG/TEST/img/*png')
            # self._mask_names= glob.glob('/scratch/yy3u19/DataSets/CRAG/TEST/labelcol/*png')

        else:
            raise RuntimeError('image set should only be train or set')
        # self._image_names.sort()
        # self._mask_names.sort()
        self.MEAN = (0.8568, 0.7219, 0.8302)
        self.STD  = (0.0935, 0.1676, 0.1277)

        self.transforms  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])
        if self.sample_weight=='sample_mean':
            self.sample_weight = torch.ones(len(self._image_names),1)/len(self._image_names)

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        # print(self._image_names[index], self._mask_names[index])
        assert(self._image_names[index].split('/')[-1]==self._mask_names[index].split('/')[-1]), self._image_names[index].split('/')[-1]+'--'+self._mask_names[index].split('/')[-1]

        image = cv2.imread('/scratch/yy3u19/DataSets/'+self._image_names[index], flags=1)
        mask  = (cv2.imread('/scratch/yy3u19/DataSets/'+self._mask_names[index], flags=0)>1).astype(np.uint8)
        mask  = mapping(mask, self.mask_channel)

        img  = self.transforms(image)
        mask = torch.from_numpy(mask).type(torch.float32)

        if self.sample_weight==None:
            return {
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
            }
        else:
            return {
                # 'image': torch.from_numpy(image).type(torch.FloatTensor),
                'image': img,
                'mask': mask,
                'name': self._image_names[index].split('/')[-1],
                'weight': self.sample_weight[index]
            }

def get_mean_std(loader):
    channels_sum, channels_squares_sum, num_batches = 0,0,0
    for data in loader:
        data = data['image']
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squares_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches +=1
    mean = channels_sum/num_batches
    std = (channels_squares_sum/num_batches-mean**2)**0.5
    return mean, std

if __name__ == '__main__':
    # train_img_pathes  = glob.glob(f'/ssd/Digest/input_cn_non_overlap_train/*jpg')+glob.glob(f'/ssd/Digest/input_cn_non_overlap_val/*jpg')
    # train_mask_pathes = glob.glob(f'/ssd/Digest/mask_cn_non_overlap_train/*jpg')+glob.glob(f'/ssd/Digest/mask_cn_non_overlap_val/*jpg')

    # train_img_pathes.sort()
    # train_mask_pathes.sort()

    # train_dataset = Digest(train_img_pathes, train_mask_pathes, sample_weight=None, mask_channel=5)
    # train_dataset = Kumar(image_set='train')
    # train_dataset = BCSS(image_set='train')
    train_dataset = CRAG(image_set='train')

    print(train_dataset.__len__())
    train_loader = DataLoader(train_dataset, batch_size=4)
    mean, std = get_mean_std(train_loader)
    print(mean)
    print(std)
