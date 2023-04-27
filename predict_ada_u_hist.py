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

def predict_img(model, net, device, dataset, alphas, level, classes, vote=None):

    print('-------', level,'-----',heads)
    save_patches = 'data/'+dataset+'/'+model+'_'+str(level)+'/'
    save_error_map = 'data/'+dataset+'/'+model+'_'+str(level)+'/error_map/'

    if not os.path.exists(save_patches):
        os.makedirs(save_patches)
    if not os.path.exists(save_error_map):
        os.makedirs(save_error_map)

    net.eval()
    if dataset=='BCSS':
        test_dataset  = BCSS('test', sample_weight='sample_mean', mask_channel=classes)
    elif dataset=='CRAG':
        test_dataset  = CRAG('test', sample_weight='sample_mean', mask_channel=classes)

    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    count=0
    miou_averaged=0
    dice_avg = 0
    metrics = Metrics(classes)
    IMG_Metrics=Metrics(classes)
    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            names = batch['name']

            output = net(imgs)
            preds = 0
            if len(level)==1:
                preds = F.softmax(output,dim=1)
            else:
                # hard voating
                if vote=='hard':
                    for i in range(0, len(alphas)):
                        batch_prob = F.softmax(output[i],dim=1)
                        max_prob= torch.max(batch_prob, dim=1,keepdim=True)[0]
                        batch_pred = (batch_prob==max_prob)
                        preds += batch_pred*alphas[i]
                # soft voating
                elif vote=='soft':
                    for i in range(0, len(alphas)): 
                        batch_prob = F.softmax(output[i],dim=1)
                        preds += batch_prob*alphas[i]
                elif vote =='fuse':
                    for i in range(0, len(alphas)):
                        batch_f = output[i]
                        preds += batch_f*alphas[i]
                    preds = F.softmax(preds,dim=1)
                elif vote=='mean':
                    for i in range(0, len(alphas)):
                        batch_prob = F.softmax(output[i],dim=1)
                        preds += batch_prob

            # torch_masks = torch.argmax(masks,dim=1).squeeze(1).view(-1)
            # torch_preds = torch.argmax(preds,dim=1).squeeze(1).view(-1)
            # metrics.add(torch_preds, torch_masks)

            torch_masks = torch.argmax(masks,dim=1)
            torch_preds = torch.argmax(preds,dim=1)
            metrics.add(torch_preds.squeeze(1).view(-1), torch_masks.squeeze(1).view(-1))

            # preds = preds.cpu().numpy()
            # masks = np.argmax(masks.cpu().numpy(), axis=1).squeeze()
            # preds = np.argmax(preds, axis=1).squeeze()

            # if len(masks.shape)==2: 
            #     masks=np.expand_dims(masks,axis=0)
            #     preds=np.expand_dims(preds,axis=0)

            # for i in range(preds.shape[0]):
            #     heatmap = (preds[i]*(256/4)).astype(np.uint8)
            #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            #     cv2.imwrite(save_patches+'/'+names[i], heatmap)

    # miou = metrics.iou(average=False)
    # print(f'fold_{test_fold}, branch:{branch}, vote:{vote}, mean: {miou.mean()}')
    # print(miou)
    cm = metrics._confusion_matrix
    print(f'branch:{level}, vote:{vote}, mean: {metrics.iou(average=True)}, mean: {metrics.iou(average=False)}')
    # get_confusion_matrix(cm.cpu().numpy(), 'ada', test_fold, level)

def get_confusion_matrix(cm, model, fold, branch):
    classes=28
    labels = [str(i) for i in range(classes)]
    acc = np.sum(cm*np.eye(classes))/np.sum(cm)
    # print('Globale acc:',acc)
    # confusion_matrix_1 = cm/np.sum(cm,axis=0)
    # confusion_matrix_2 = cm/np.sum(cm)

    xlocations = np.array(range(len(labels)))
    plt.figure(figsize=(18,19))
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
    plt.savefig(f'{model}_{fold}_{branch}_cm.png')

    # plt.figure(figsize=(9,9))
    # plt.xticks(xlocations, labels,fontsize=10)
    # plt.yticks(xlocations, labels,fontsize=10)
    # plt.title(model+', Accuracy: '+'%.3f' %(acc*100)+'%', family='fantasy', fontsize=15)
    # plt.ylabel('True label', fontsize=10)
    # plt.xlabel('Predicted label', fontsize=10)
    # plt.tight_layout()
    # x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    # for x,y in zip(x.flatten(), y.flatten()):
    #     prob = confusion_matrix_2[x,y]
        # if x==y:
        #     plt.text(x,y,"%00.2f" %(prob*100,), color='green', fontsize=10, va='center', ha='center')
    # plt.imshow(confusion_matrix_2, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.savefig(model+'_'+fold+'cm_2.png')

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
    n_classes = 5
    filters = 64
    n_channels = 3
    f=open(args.weight+'.txt','r')
    alphas=[]
    heads =[]
    i=0
    for line in f:
        if i==4:
            break
        alphas.append(np.float(line))
        i+=1 
    if len(args.level) != 1:
        net = AdaBoost_UNet(n_channels=n_channels, n_classes=n_classes, level='1234_pred',
            filters=filters, skip_option=args.skip, deep_sup=False).cuda()
    else:
        net = AdaBoost_UNet(n_channels=n_channels, n_classes=n_classes, level=str(args.level),
            filters=filters, skip_option=args.skip, deep_sup=False).cuda()

    # args.level = heads
    print(args.skip, alphas, args.level, args.vote)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Loading model {}".format(args.weight))
    logging.info(f'Using device {device}')
    # summary(net, (1, 512, 512))
    count=0
    net.to(device=device)
    net.load_state_dict(torch.load(args.weight+'.pth', map_location=device))

    # num=0
    # for name,para in net.named_parameters():
    #         if 'up_11' in name or 'up_21' in name or 'up_12' in name or 'up_31' in name or 'up_22' in name or 'up_13' in name or 'out_02' in name or 'out_12' in name or 'out_03' in name or 'out_04' in name  or 'out_22' in name or 'out_13' in name:
    #             continue
    #         else:
    #             print(name)
    #             temp = 1
    #             size = para.size()
    #             s=1
    #             for d in size:
    #                 s *= d
    #             num += s
    # print('num of paras:', num)

    predict_img(args.model, net, device, 'BCSS', alphas, args.level, 5, vote=args.vote)
