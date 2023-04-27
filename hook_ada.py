import os
import sys
import glob
import torch
import pwcca
import numpy as np
import torch.nn as nn
import cca_core
from utils.dataset import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.data import DataLoader
from unet.Adaboost_UNet import AdaBoost_UNet

gpu = sys.argv[1]
fold= sys.argv[2]
skip= sys.argv[3]

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
state_dict = torch.load(f'checkpoints/kylberg/30-epoch/bce_{fold}_0.001_True_scse_DS.pth', map_location=device)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
    # name = k[7:] # remove `module.`
    # new_state_dict[name] = v
# load params
classes = 28
nodes = 9
model =  AdaBoost_UNet(n_channels=1, n_classes=classes, filters=16, skip_option=skip, level='outer').cuda()

model.to(device=device)
# model.load_state_dict(new_state_dict)
model.load_state_dict(state_dict)

test_imgs = np.array(glob.glob(f'/ssd/kylberg/fold_{fold}/imgs/*'))
test_masks = np.array(glob.glob(f'/ssd/kylberg/fold_{fold}/gt/*'))

idx = [i for i in range(len(test_imgs))]
# print(idx)
np.random.seed(2)
np.random.shuffle(idx)
test_imgs = test_imgs[idx][:500]
test_masks = test_masks[idx][:500]

val_dataset   = CustomImageSet(test_imgs, test_masks, sample_weight='sample', mask_channel=28)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=6, pin_memory=True)

features = [torch.empty([0,16*(2**i),32,32]).to(device=device, dtype=torch.float32) for i in range(0,5)]+\
            [torch.empty([0,128,32,32]).to(device=device, dtype=torch.float32),
             torch.empty([0,64,32,32]).to(device=device, dtype=torch.float32),
             torch.empty([0,32,32,32]).to(device=device, dtype=torch.float32),
             torch.empty([0,16,32,32]).to(device=device, dtype=torch.float32)]

for f in features:
    print(f.size())
model.eval()
for batch in val_loader:
    imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
    outputs = model(imgs)
    for i in range(nodes):
        if i<4:
            features[i] = torch.cat([features[i],F.avg_pool2d(outputs[i].detach(),kernel_size=2**(4-i), stride=2**(4-i))],dim=0)
        elif i ==4:
            features[i] = torch.cat([features[i],outputs[i].detach()],dim=0)
        else:
            out = F.avg_pool2d(outputs[i].detach(),kernel_size=2**(i-4), stride=2**(i-4))
            features[i] = torch.cat([features[i], out],dim=0)
        # print(features[i].size())

print('----------------------------')

for i in range(nodes):
    features[i] = features[i].cpu().numpy()
    features[i] = np.transpose(features[i],[0,2,3,1])
    features[i] = np.reshape(features[i],[-1,features[i].shape[3]])
    features[i] = np.transpose(features[i],[1,0])
    print(features[i].shape)

svcca = np.zeros([nodes,nodes])
for i in range(nodes):
    f_1 = features[i]
    for j in range(i,nodes):
        f_2 = features[j]
        print(f'f_1 shape:{f_1.shape}, f_2 shape:{f_2.shape}')    
        if f_1.shape[0]<f_2.shape[0]:
            pwcca_mean, w, _ = pwcca.compute_pwcca(f_1, f_2, epsilon=1e-10)
        else:
            pwcca_mean, w, _ = pwcca.compute_pwcca(f_2, f_1, epsilon=1e-10)
        # a_results = cca_core.get_cca_similarity(f_1, f_2, epsilon=1e-8, verbose=False)["cca_coef1"]
        # pwcca_mean = np.mean(a_results[:16])

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
    labels = ['X_00','X_10','X_20','X_30','X_40','X_31','X_22','X_13','X_04']
    plt.figure(figsize=(12, 12))
    plt.xticks(xlocations, labels,fontsize=13)
    labels.reverse()
    plt.yticks(xlocations, labels,fontsize=13)
    plt.title('PWCCA of '+model, fontsize=15)
    plt.ylabel('layers', fontsize=15)
    plt.xlabel('layers', fontsize=15)
    plt.tight_layout()
    x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    for i in range(nodes):
        for j in range(nodes):
            plt.text(j,i,"%00.2f" %(similarity_score[i,j]), color='black', fontsize=14, va='center', ha='center')
    plt.imshow(similarity_score, cmap=plt.cm.viridis, vmin=0.0, vmax=1.0)
    # plt.colorbar()
    # plt.show()
    plt.savefig(model+'_svccca.png')
    return similarity_score

get_similarity_matrix(svcca, f'Adaboost_UNet 9({skip})')
