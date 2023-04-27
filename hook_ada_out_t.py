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

def get_cca(gpu, fold):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(f'checkpoints/kylberg/30-epoch/bce_{fold}_0.001_True_scse_DS.pth', map_location=device)

    classes = 28
    nodes = 4
    model =  AdaBoost_UNet(n_channels=1, n_classes=classes, filters=16, skip_option='scse', level='1234_feat').cuda()

    model.to(device=device)
    model.load_state_dict(state_dict)

    test_imgs = np.array(glob.glob(f'/ssd/kylberg/fold_{fold}/imgs/*'))
    test_masks = np.array(glob.glob(f'/ssd/kylberg/fold_{fold}/gt/*'))
    val_dataset   = CustomImageSet(test_imgs, test_masks, sample_weight='sample', mask_channel=28)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=6, pin_memory=True)

    features = [torch.empty([0,16,32,32]).to(device=device, dtype=torch.float32) for i in range(0,4)]

    for f in features:
        print(f.size())
    model.eval()
    for batch in val_loader:
        imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
        outputs = model(imgs)
        for i in range(nodes):
            features[i] = torch.cat([features[i],F.avg_pool2d(outputs[i].detach(),kernel_size=16, stride=16)],dim=0)

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
        # Perform SVD
        # U1, s1, V1 = np.linalg.svd(f_1, full_matrices=False)
        # print(f'f1:{f_1.shape}, U1:{U1.shape}, s1:{s1.shape}, V1:{V1.shape}')
        # f_1 = np.dot(s1[:16]*np.eye(16), V1[:16])
        for j in range(i,nodes):
            f_2 = features[j]
            # U2, s2, V2 = np.linalg.svd(f_2, full_matrices=False)
            # f_2 = np.dot(s2[:16]*np.eye(16), V2[:16])
            print(f'f_1 shape:{f_1.shape}, f_2 shape:{f_2.shape}')
            if f_1.shape[0]<f_2.shape[0]:
                pwcca_mean, w, _ = pwcca.compute_pwcca(f_1, f_2, epsilon=1e-10)
            else:
                pwcca_mean, w, _ = pwcca.compute_pwcca(f_2, f_1, epsilon=1e-10)

            print(f'{i}-{j}: {pwcca_mean}')
            svcca[i,j] = pwcca_mean
            svcca[j,i] = pwcca_mean

    print(f'--------------fold:{fold}------------------------')
    print(svcca)
    return svcca

def get_similarity_matrix(similarity_score, model):
#     print(similarity_score)
    similarity_score= np.rot90(similarity_score,1)
    print(similarity_score)
    nodes = 4
    xlocations = np.array([i for i in range(nodes)])
    labels = ['X_01','X_02','X_03','X_04']
    plt.figure(figsize=(6, 6))
    plt.xticks(xlocations, labels,fontsize=20)
    labels.reverse()
    plt.yticks(xlocations, labels,fontsize=20)
    # plt.title('PWCCA of '+model, fontsize=20)
    plt.tight_layout()
    x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    for i in range(nodes):
        for j in range(nodes):
            plt.text(j,i,"%00.3f" %(similarity_score[i,j]), color='black', fontsize=21, va='center', ha='center')
    plt.imshow(similarity_score, cmap=plt.cm.viridis, vmin=0.0, vmax=1.0)
    # plt.colorbar()
    # plt.show()
    plt.savefig(model+'_pwcca.png',bbox_inches='tight')
    return similarity_score

svcca = np.zeros([4,4])
for i in [0,2,3,4]:
    svcca += get_cca(0, i)
get_similarity_matrix(svcca/4, f'Ada_DS_UNet_out')

