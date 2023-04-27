from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

class Prague(Dataset):
    def __init__(self, img_pth, mask_pth):   # initial logic happens like transform
        self.image_paths = img_pth
        self.mask_paths = mask_pth

    def __getitem__(self, index):
        # print(self.image_paths[index])
        image = cv2.imread((self.image_paths[index]),flags=1)  # image BGR
        image = np.transpose(image,[2,0,1])
        # image = image/255..0
        # image = self.to_tensor(image).float()
        # image = self.transforms(image)
        return image


    def __len__(self):  # return count of sample we have
        return len(self.image_paths)

img_path = glob.glob('/ssd/kylberg/train-imgs/*')
mask_path = glob.glob('/ssd/kylberg/train-gts/*')
# print(img_path)
img_path.sort()
mask_path.sort()

dataset = Prague(img_path, mask_path)

loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)


mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1) /255.0
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print('mean:', mean, 'std:', std)