import os
import cv2
import glob
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from unet import UNet
from metrics import *
from utils.dataset import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchsummary import summary
from unet.unet_pp import UNet2Plus
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from unet.Adaboost_UNet import AdaBoost_UNet
# from unet.Adaboost_UNet_1 import AdaBoost_UNet_1 as AdaBoost_UNet

def predict_img(model, net, device, dataset, alphas, level, heads, vote=None):
    print('-------', level,'-----',heads)
    save_patches = 'data/'+dataset+'/'+model+'_'+str(level)+'/'
    if not os.path.exists(save_patches):
        os.makedirs(save_patches)

    net.eval()

    test_dataset  = PAIP(image_set='test', sample_weight='sample_mean')
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    count=0
    miou_averaged=0
    metrics = Metrics(2)
    IMG_Metrics=Metrics(2)
    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            names = batch['name']

            preds = 0
            if not isinstance(level, list):
                output, eta = net(imgs)
                eta = normalize(eta)
                node = torch.argmax(eta)
                # print(f'max eta:{eta.max()}, idx:{node}')
                preds = F.interpolate(torch.sigmoid(output[node]), size=(512,512), mode='bilinear', align_corners=True)
            else:
                # hard voating
                output_dict = net(imgs)
                if vote=='hard':
                    for i in range(1, len(alphas)+1):
                        batch_prob = F.softmax(output[i],dim=1)
                        max_prob= torch.max(batch_prob, dim=1,keepdim=True)[0]
                        batch_pred = (batch_prob==max_prob)
                        preds += batch_pred*alphas[i]
                # soft voating
                elif vote=='soft':
                    for i in range(0, len(alphas)):
                        eta, output = output_dict[i+1]
                        eta = normalize(eta)
                        node = torch.argmax(eta)
                        preds += F.interpolate(torch.sigmoid(output[node]), size=(512,512), mode='bilinear', align_corners=True)*alphas[i]
                elif vote=='mean':
                    for i in range(0, len(alphas)):
                        eta, output = output_dict[i+1]
                        eta = normalize(eta)
                        alphas = [0.25, 0.25, 0.25, 0.25]
                        node = torch.argmax(eta)
                        preds += F.interpolate(torch.sigmoid(output[node]), size=(512,512), mode='bilinear', align_corners=True)*alphas[i]

            torch_masks = (masks>=0.5).squeeze(1)
            torch_preds = (preds>=0.5).squeeze(1)
            metrics.append_iou(torch_preds, torch_masks)
            metrics.add(torch_preds.view(-1), torch_masks.view(-1))

            preds = torch_preds.cpu().numpy()
            masks = torch_masks.cpu().numpy()

            for i in range(preds.shape[0]):
                temp_miou, temp_iou_mask, iou = IMG_Metrics.get_img_iou(torch_preds.view(-1), torch_masks.view(-1))
                heatmap = preds[i]*255
                heatmap = heatmap.astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(save_patches+'/'+names[i].split('/')[-1][:-4]+'_'+str(round(temp_miou,4))+str(temp_iou_mask)+'_'+str(iou)+'.png', heatmap)
                miou_averaged += temp_miou
                count+=1
    # miou = metrics.iou(average=False)
    # print(f'fold_{test_fold}, branch:{branch}, vote:{vote}, mean: {miou.mean()}')
    # print(miou)
    miou_list = metrics._iou_list
    cm = metrics._confusion_matrix
    print(f'branch:{level}, vote:{vote}, mean: {metrics.iou(average=False).mean()}, {miou_list.mean()}')
    print( metrics.iou(average=False))
    get_confusion_matrix(cm.cpu().numpy(), model, level)

def normalize(eta):
    eta = eta.view(-1)
    epslion = 1/len(eta)
    eta = F.softmax(eta, dim=0)+epslion
    eta = eta/eta.sum()
    return eta

def get_confusion_matrix(cm, model, branch):
    classes=2
    labels = [str(i) for i in range(classes)]
    acc = np.sum(cm*np.eye(classes))/np.sum(cm)
    # print('Globale acc:',acc)
    # confusion_matrix_1 = cm/np.sum(cm,axis=0)
    # confusion_matrix_2 = cm/np.sum(cm)

    xlocations = np.array(range(len(labels)))
    plt.figure(figsize=(8,9))
    plt.xticks(xlocations, labels,fontsize=12)
    plt.yticks(xlocations, labels,fontsize=12)
    plt.title(model+', Accuracy: '+'%.3f' %(acc*100)+'%', family='fantasy', fontsize=15)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    cm = cm/np.sum(cm,axis=1)
    x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    for x,y in zip(x.flatten(), y.flatten()):
        prob = cm[x,y]
        if x==y:
            plt.text(x,y,"%00.1f" %(prob*100,), color='green', fontsize=12, va='center', ha='center')
    plt.imshow(cm, cmap=plt.cm.jet)
    plt.colorbar()
    plt.savefig(f'{model}_{branch}_cm.png')

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, metavar='FILE', default='ada',
                        help="Specify the network name")
    parser.add_argument('--weight', '-w', type=str, metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-o', '--skip', dest='skip', type=str, default='scse',
                        help='skip'),
    parser.add_argument('--level', '-l', metavar='FILE', type=str, default='1')
    parser.add_argument('--gpu', '-g', metavar='FILE', type=str, default='0',
                        help="Specify the file in which the model is stored")
    # parser.add_argument('--branch', '-b', dest='branch', type=int, help='branch')
    parser.add_argument('--vote', '-v', dest='vote', type=str, default='soft')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.set_device(0)
    f=open(args.weight+'.txt','r')
    alphas=[]
    heads =[]
    i=0
    for line in f:
        if i==4:
            break
        head, alpha = line.split(' ')
        alphas.append(np.float(alpha))
        heads.append(int(head))
        i+=1 
    alphas = np.array(alphas)
    alphas = alphas/alphas.sum()
    if len(args.level) != 1:
        args.level=heads
        net = AdaBoost_UNet(n_channels=3, n_classes=2, level=args.level,
            filters=16, skip_option=args.skip, deep_sup=True).cuda()
    else:
        net = AdaBoost_UNet(n_channels=3, n_classes=2, level=str(args.level),
            filters=16, skip_option=args.skip, deep_sup=True).cuda()

    # args.level = heads
    print(args.skip, alphas, args.level, args.vote)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Loading model {}".format(args.weight))
    logging.info(f'Using device {device}')
    # summary(net, (1, 512, 512))
    count=0
    net.to(device=device)
    net.load_state_dict(torch.load(args.weight+'.pth', map_location=device))

    predict_img(args.model, net, device, 'PAIP', alphas, args.level, heads=heads, vote=args.vote)
