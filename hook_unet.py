import os
import sys
import glob
import torch
import pwcca
import cca_core
import numpy as np
import torch.nn as nn
from unet import UNet
from utils.dataset import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.data import DataLoader
from unet.deep_supervise_unet_model import DS_UNet_1 as DS_UNet
from cka_sim import *

print('---------------------ds_unet---++------------------')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# state_dict = torch.load(f'checkpoints/BCSS/UNet/34_0.001_UNet_ce.pth', map_location=device)
state_dict = torch.load(f'checkpoints/BCSS/DSU/34_0.001_DS_ce_max.pth', map_location=device)
# state_dict = torch.load(f'checkpoints/BCSS/cascade_sample_weight/ce_0.002_Ada_scse.pth', map_location=device)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = 'module.'+k # remove `module.`
#     new_state_dict[name] = v
# load params
classes = 5
nodes = 8
# model = UNet(n_channels=3, n_classes=classes, filters=64, flag=True).cuda()
model = DS_UNet(n_channels=3, n_classes=classes, filters=64).cuda()

model.to(device=device)
# model.load_state_dict(new_state_dict)
model.load_state_dict(state_dict)

test_dataset  = BCSS('test', sample_weight=None, mask_channel=5)
val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=6, pin_memory=True)

channels = [128, 256, 512, 1024, 512, 256, 128, 64]
size     = [8,   4,   2,   0,    2,   4,   8,   16]
features = [torch.empty([0,c,32,32]) for c in channels]

for f in features:
    print(f.size())

model.eval()
for batch in val_loader:
    imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
    outputs = model(imgs)
    for i in range(nodes):
        output = outputs[i]
        if i== 3:
            features[i] = torch.cat([features[i],output.detach().cpu()],dim=0)
        else:
            features[i] = torch.cat([features[i],F.avg_pool2d(output.detach().cpu(),kernel_size=size[i], stride=size[i])],dim=0)
        if i%10==0:
            print(features[i].size())

print('----------------------------')

# pwcca
# for i in range(nodes):
#     features[i] = features[i].cpu().numpy()
#     features[i] = np.transpose(features[i],[0,2,3,1])
#     features[i] = np.reshape(features[i],[-1,features[i].shape[3]])
#     features[i] = np.transpose(features[i],[1,0])
#     print(features[i].shape)

# cka
for i in range(nodes):
    features[i] = features[i].cpu().numpy()
    features[i] = np.reshape(features[i],[features[i].shape[0],-1])
    print(features[i].shape)

svcca = np.zeros([nodes,nodes])
for i in range(nodes):
    f_1 = features[i]
    for j in range(i,nodes):
        f_2 = features[j]

        # pwcca
        # if f_1.shape[0]<f_2.shape[0]:
        #     pwcca_mean, w, _ = pwcca.compute_pwcca(f_1, f_2, epsilon=1e-10)
        # else:
        #     pwcca_mean, w, _ = pwcca.compute_pwcca(f_2, f_1, epsilon=1e-10)

        # CKA
        pwcca_mean =cka(gram_rbf(f_1, 0.5), gram_rbf(f_2, 0.5))

        print(f'{i}-{j}: {pwcca_mean}')
        svcca[i,j] = pwcca_mean
        svcca[j,i] = pwcca_mean

print('--------------------------------------')
print(svcca)

def get_similarity_matrix(similarity_score, model):
#     print(similarity_score)
    similarity_score= np.rot90(similarity_score,1)
    print(similarity_score)

    xlocations = np.array([i for i in range(nodes)])
    labels = ['X_10','X_20','X_30','X_40','X_31','X_22','X_13','X_04']
    plt.figure(figsize=(21,21))
    plt.xticks(xlocations, labels,fontsize=20)
    labels.reverse()
    plt.yticks(xlocations, labels,fontsize=20)
    # plt.title('PWCCA of '+model, fontsize=20)
    plt.tight_layout()
    x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    for i in range(nodes):
        for j in range(nodes):
            plt.text(j,i,"%00.3f" %(similarity_score[i,j]), color='black', fontsize=21, va='center', ha='center')
    plt.imshow(similarity_score, cmap=plt.cm.hot, vmin=0.0, vmax=1.0)
    # plt.colorbar()
    # plt.show()
    plt.savefig(model+'_pwcca.png',bbox_inches='tight')
    return similarity_score

get_similarity_matrix(svcca, f'DS_UNet')
