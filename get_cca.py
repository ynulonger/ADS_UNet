import os
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
from unet.Adaboost_UNet import AdaBoost_UNet
from unet.deep_supervise_unet_model import DS_UNet as DS_UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 1, 3"
classes = 28
nodes = 9
imgs = 52
device_1 = torch.device('cuda:0')
state_dict = torch.load('checkpoints/Kylberg/UNet_4.pth', map_location=device_1)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model_1 =  UNet(n_channels=1, n_classes=classes, filters=16, bilinear=False).cuda()
model_1.to(device=device_1)
model_1.load_state_dict(new_state_dict)

device_2 = torch.device('cuda:1')
state_dict = torch.load('checkpoints/Kylberg/iou_Ada_update_weight/Ada_iou_scse_1.pth', map_location=device_2)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model_2 =  AdaBoost_UNet(n_channels=1, n_classes=classes, level='123456789', filters=16, skip_option='scse').cuda()
model_2.to(device=device_2)
model_2.load_state_dict(new_state_dict)

device_3 = torch.device('cuda:2')
state_dict = torch.load('checkpoints/Kylberg/avg_DS_UNet/DS_UNet_4.pth', map_location=device_3)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model_3 = DS_UNet(n_channels=1, n_classes=classes, filters=16, bilinear=False).cuda()
model_3.to(device=device_3)
model_3.load_state_dict(new_state_dict)

test_imgs = glob.glob(f'/home/yy3u19/mycode/Kylberg/fold_4/imgs/*')
np.random.shuffle(test_imgs)
test_imgs = test_imgs[:imgs]

val_dataset   = ImageSet(test_imgs, sample_weight=None, mask_channel=28)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

features_1 = [torch.empty([0,16*(2**i), 512//(2**i), 512//(2**i)]).to(device=device_1, dtype=torch.float32) for i in range(0,5)]+ [torch.empty([0,128,64,64]).to(device=device_1, dtype=torch.float32),
             torch.empty([0,64,128,128]).to(device=device_1, dtype=torch.float32),
             torch.empty([0,32,256,256]).to(device=device_1, dtype=torch.float32),
             torch.empty([0,16,512,512]).to(device=device_1, dtype=torch.float32)]
features_2 = [torch.empty([0,16*(2**i), 512//(2**i), 512//(2**i)]).to(device=device_2, dtype=torch.float32) for i in range(0,5)]+ [torch.empty([0,128,64,64]).to(device=device_2, dtype=torch.float32),
             torch.empty([0,64,128,128]).to(device=device_2, dtype=torch.float32),
             torch.empty([0,32,256,256]).to(device=device_2, dtype=torch.float32),
             torch.empty([0,16,512,512]).to(device=device_2, dtype=torch.float32)]
features_3 = [torch.empty([0,16*(2**i), 512//(2**i), 512//(2**i)]).to(device=device_3, dtype=torch.float32) for i in range(0,5)]+ [torch.empty([0,128,64,64]).to(device=device_3, dtype=torch.float32),
             torch.empty([0,64,128,128]).to(device=device_3, dtype=torch.float32),
             torch.empty([0,32,256,256]).to(device=device_3, dtype=torch.float32),
             torch.empty([0,16,512,512]).to(device=device_3, dtype=torch.float32)]
model_1.eval()
model_2.eval()
model_3.eval()

for batch in val_loader:
    imgs_1 = Variable(batch['image'].to(device=device_1, dtype=torch.float32))
    outputs_1 = model_1(imgs_1)
    imgs_2 = Variable(batch['image'].to(device=device_2, dtype=torch.float32))
    outputs_2 = model_2(imgs_2)
    imgs_3 = Variable(batch['image'].to(device=device_3, dtype=torch.float32))
    outputs_3 = model_3(imgs_3)

    for i in range(nodes):
        features_1[i] = torch.cat([features_1[i],outputs_1[i].detach()],dim=0)
        features_2[i] = torch.cat([features_2[i],outputs_2[i].detach()],dim=0)
        features_3[i] = torch.cat([features_3[i],outputs_3[i].detach()],dim=0)

print('----------------------------')

for i in range(nodes):
    print(features_1[i].size())
    features_1[i] = features_1[i].cpu().numpy()
    features_1[i] = np.transpose(features_1[i],[1,0,2,3])
    features_1[i] = np.reshape(features_1[i],[imgs,-1])

    features_2[i] = features_2[i].cpu().numpy()
    features_2[i] = np.transpose(features_2[i],[1,0,2,3])
    features_2[i] = np.reshape(features_2[i],[imgs,-1])

    features_3[i] = features_3[i].cpu().numpy()
    features_3[i] = np.transpose(features_3[i],[1,0,2,3])
    features_3[i] = np.reshape(features_3[i],[imgs,-1])
    # features[i] = np.transpose(features[i],[1,0])

    # print(features_1[i].shape)

def _plot_helper(arr_1, arr_2, arr_3, title):
    x= [i for i in range(9)]
    plt.title(title)
    plt.xlabel('nodes')
    plt.ylabel('PWCCA similarity')
    plt.xticks(x, ['X_00','X_10','X_20','X_30','X40','X_31','X_22','X_13','X_04'])
    plt.grid()

    A,=plt.plot(arr_1, lw=1.0, marker='+', label='UNet VS Ada-Net')
    B,=plt.plot(arr_2, lw=1.0, marker='*', label='UNet VS DSU-UNet')
    C,=plt.plot(arr_3, lw=1.0, marker='o', label='DSU-Net VS Ada-UNet')
    legend = plt.legend(handles=[A,B,C])
    plt.savefig('PWCCA_Kylberg.png')

def get_similarity_curve(DSU_list, UNet_list):
    similarity_score = []
    for dsu, unet in zip(DSU_list, UNet_list):
        similarity, w, c = pwcca.compute_pwcca(dsu, unet, epsilon=1e-10)
        print(similarity)
        similarity_score.append(similarity)
    similarity_score=np.array(similarity_score)
    return similarity_score

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
    plt.title('svcca of '+model, fontsize=15)
    plt.ylabel('layers', fontsize=13)
    plt.xlabel('layers', fontsize=13)
    plt.tight_layout()
    x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    for i in range(nodes):
        for j in range(nodes):
            plt.text(j,i,"%00.2f" %(similarity_score[i,j]), color='black', fontsize=12, va='center', ha='center')
    plt.imshow(similarity_score, cmap=plt.cm.jet, vmin=0.0, vmax=1.0)
    plt.colorbar()
    # plt.show()
    plt.savefig(model+'_svccca.png')
    return similarity_score

# get_similarity_matrix(svcca, 'UNet')
s_1=get_similarity_curve(features_1, features_2)
s_2=get_similarity_curve(features_1, features_3)
s_3=get_similarity_curve(features_2, features_3)

print(s_1)
print(s_2)
print(s_3)
_plot_helper(s_1, s_2, s_3, 'Texture Dataset')