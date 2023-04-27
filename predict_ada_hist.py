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

def predict_img(model, net, device, dataset, alphas, level, heads, classes, vote=None):
    print('-------', level,'-----',heads)
    save_patches = 'data/'+dataset+'/'+model+'_'+str(level)+'_'+vote+'/'
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
    elif dataset=='Kumar':
        test_dataset  = Kumar('test', sample_weight='sample_mean', mask_channel=classes)

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
            # print('img size:', imgs.size())
            preds = 0
            if not isinstance(level, list):
                # print(len(output), heads[int(level)-1])
                output, eta = net(imgs)
                eta = normalize(eta)
                # print(eta)
                # ensemble
                '''
                for i in range(0,len(eta)):
                    preds += F.interpolate(F.softmax(output[i],dim=1), size=(imgs.size(2),imgs.size(3)), mode='bilinear', align_corners=True)
                '''
                node = torch.argmax(eta)
                preds = F.interpolate(F.softmax(output[heads[int(level)-1]],dim=1), size=(imgs.size(2),imgs.size(3)), mode='bilinear', align_corners=True)

            else:
                # hard voating
                output_dict = net(imgs)
                if vote=='hard':
                    for i in range(1, len(alphas)+1):
                        batch_prob = F.softmax(output[i],dim=1)
                        max_prob= torch.max(batch_prob, dim=1,keepdim=True)[0]
                        batch_pred = (batch_prob==max_prob)
                        preds += batch_pred*alphas[i]
                # soft voting
                elif vote=='soft':
                    for i in range(0, len(alphas)):
                        eta, output = output_dict[i+1]
                        eta = normalize(eta)
                        # print(eta)
                        branch_pred = 0
                        # ensemble
                        
                        # for j in range(0,len(eta)):
                        #     branch_pred += eta[i]*F.interpolate(F.softmax(output[j],dim=1), size=(imgs.size(2),imgs.size(3)), mode='bilinear', align_corners=True)

                        branch_pred = F.interpolate(F.softmax(output[heads[i]],dim=1), size=(imgs.size(2),imgs.size(3)), mode='bilinear', align_corners=True)
                        preds += branch_pred*alphas[i]

                elif vote=='mean':
                    for i in range(0, len(alphas)):
                        eta, output = output_dict[i+1]
                        eta = normalize(eta)
                        # print(eta)
                        branch_pred = 0
                        for j in range(0,len(eta)):
                            branch_pred += F.interpolate(F.softmax(output[j],dim=1), size=(imgs.size(2),imgs.size(3)), mode='bilinear', align_corners=True)
                        # branch_pred = F.softmax(output[-1],dim=1)
                        # branch_pred = F.interpolate(F.softmax(output[heads[i]],dim=1), size=(imgs.size(2),imgs.size(3)), mode='bilinear', align_corners=True)
                        preds += branch_pred
                        
            torch_masks = torch.argmax(masks,dim=1)
            torch_preds = torch.argmax(preds,dim=1)
            # print(names, torch_preds.size(), torch_masks.size())
            # metrics.append_iou(torch_preds, torch_masks)
            metrics.add(torch_preds.squeeze(1).view(-1), torch_masks.squeeze(1).view(-1))

            # '''
            preds = preds.cpu().numpy()
            masks = np.argmax(masks.cpu().numpy(), axis=1).squeeze()
            preds = np.argmax(preds, axis=1).squeeze()

            if len(masks.shape)==2: 
                masks=np.expand_dims(masks,axis=0)
                preds=np.expand_dims(preds,axis=0)

            for i in range(preds.shape[0]):
                # print(names[i], torch.unique(torch_preds), torch.unique(torch.mask))
                img_iou = IMG_Metrics.get_img_iou(torch_preds.view(-1), torch_masks.view(-1)).item()
                dice = IMG_Metrics.multi_dice_coef(torch_preds.view(-1), torch_masks.view(-1), 5)
                print(img_iou)
            '''    
                heatmap = (preds[i]*(255/4)).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
                # pixel_error = np.sum((preds[i] != masks[i]))/(512**2)
                error_map = ((preds[i] != masks[i])*255).astype(np.uint8)
                error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
                cv2.imwrite(save_patches+'/'+names[i].split('.')[0]+'_'+str(img_iou)[:5]+'.png', heatmap)
                # cv2.imwrite(save_patches+'/'+names[i].split('.')[0]+'.png', preds[i])
                # cv2.imwrite(save_error_map+'/'+names[i].split('.')[0]+'_'+str(pixel_error)+'.png', error_map)
                count+=1
                # cv2.imwrite(save_patches+'/m_'+names[i], preds[i])
                # print(temp_iou)
                dice_avg += dice
            '''
    print(f'branch:{level}, vote:{vote}, miou:{metrics.iou(average=True)}, dice:{metrics.dice(average=True)},')

    cm = metrics._confusion_matrix
    print(metrics.iou(average=True), metrics.iou(average=False))
    # print(metrics._confusion_matrix)
    # get_confusion_matrix(cm.cpu().numpy(), 'ada_'+str(level), 'data/'+dataset+'/', level)

def normalize(eta):
    eta = eta.view(-1)
    epslion = 1/len(eta)
    eta = F.softmax(eta, dim=0)+epslion
    eta = eta/eta.sum()
    return eta

def get_confusion_matrix(cm, model, save_dir, branch):
    classes=5
    labels = [str(i) for i in range(classes)]
    acc = np.sum(cm*np.eye(classes))/np.sum(cm)
    print('Globale acc:',acc)
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
    plt.savefig(f'{save_dir}{model}_{branch}_cm.png')

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
    parser.add_argument('--filters', '-f', metavar='FILE', type=int, default=64,
                        help="Specify the file in which the model is stored")
    parser.add_argument('-d', '--data', dest='data', type=str, default='BCSS',
                        help='dataset')
    # parser.add_argument('--branch', '-b', dest='branch', type=int, help='branch')
    parser.add_argument('--vote', '-v', dest='vote', type=str, default='soft')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.data=='BCSS':
        classes = 5
    else:
        classes = 2

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
    if len(args.level) != 1:
        args.level=heads
        net = AdaBoost_UNet(n_channels=3, n_classes=classes, level=args.level,
            filters=args.filters, skip_option=args.skip, deep_sup=True).cuda()
    else:
        net = AdaBoost_UNet(n_channels=3, n_classes=classes, level=str(args.level),
            filters=args.filters, skip_option=args.skip, deep_sup=True).cuda()

    # args.level = heads
    print('-------',args.data,args.model,args.skip, alphas, args.level, args.vote,'-------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Loading model {}".format(args.weight))
    logging.info(f'Using device {device}')
    # summary(net, (3, 512, 512))
    count=0
    net.to(device=device)
    net.load_state_dict(torch.load(args.weight+'.pth', map_location=device))

    predict_img(args.model, net, device, args.data, alphas, args.level, heads=heads, classes=classes, vote=args.vote)
