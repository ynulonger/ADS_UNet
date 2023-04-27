import os
import cv2
import glob
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from unet import UNet
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

def predict_img(model, net, device, fold, alphas, branch):
    batch_size=2
    save_patches = 'data/Kylberg/'+str(branch)+'_'+model+'_'+fold
    if not os.path.exists(save_patches):
        os.makedirs(save_patches)

    net.eval()

    img_pathes = glob.glob('/home/yy3u19/mycode/Kylberg/'+ fold+'/imgs/*')
    mask_pathes = glob.glob('/home/yy3u19/mycode/Kylberg/'+ fold+'/gt/*')
    print(len(img_pathes))
    img_pathes.sort()
    mask_pathes.sort()
    val_dataset   = CustomImageSet(img_pathes, mask_pathes, mask_channel=28)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    ground_truth = np.empty([0, 512,512])
    predictions  = np.empty([0, 512,512])

    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            masks = batch['mask']
            names = batch['name']

            output = net(imgs)
            preds = 0

            if branch<5:
                preds = F.softmax(output[branch],dim=1)
            else:
                for i in range(0, len(alphas)):
                    batch_prob = F.softmax(output[i],dim=1)
                    max_prob= torch.max(batch_prob, dim=1,keepdim=True)[0]
                    batch_pred = (batch_prob==max_prob)
                    preds += batch_pred*alphas[i]

                # for i in range(0, len(alphas)):
                #     batch_f = output[i]
                #     preds += batch_f*alphas[i]
                # preds = F.softmax(preds,dim=1)

                # for i in range(0, len(alphas)):
                #     batch_prob = F.softmax(output[i],dim=1)
                #     preds += batch_prob*alphas[i]

            preds = preds.cpu().numpy()
            masks = np.argmax(masks.cpu().numpy(), axis=1).squeeze()
            preds = np.argmax(preds, axis=1).squeeze()

            if len(masks.shape)==2: 
                masks=np.expand_dims(masks,axis=0)
                preds=np.expand_dims(preds,axis=0)

            ground_truth = np.concatenate([ground_truth, masks], axis=0)
            # print(preds.shape)
            predictions  = np.concatenate([predictions, preds], axis=0)
            for i in range(preds.shape[0]):
                heatmap = (preds[i]*(256/27)).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(save_patches+'/'+names[i], heatmap)
                cv2.imwrite(save_patches+'/m_'+names[i], preds[i])

    # print(predictions.shape, ground_truth.shape)
    # print('branch', branch)
    predictions = np.reshape(predictions, [-1])
    ground_truth= np.reshape(ground_truth,[-1])
    get_confusion_matrix(predictions, ground_truth, model, fold, branch)
    # mean_iou = iou_pytorch(predictions, ground_truth)
    # print('m_IoU:', mean_iou)

def iou_pytorch(outputs, labels):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    SMOOTH = 1e-6
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    # labels = labels.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    iou = 0
    for  i in range(3):
        intersection = ((outputs==i) & (labels==i)).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = ((outputs==i) | (labels==i)).float().sum((1, 2))         # Will be zzero if both are 0
        iou_temp = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
        iou +=iou_temp
    iou =iou/3
    mean_iou = torch.sum(iou).item()
    return mean_iou

def get_confusion_matrix(prediction, truth, model, fold, branch):
    classes=28
    labels = [str(i) for i in range(classes)]

    cm = confusion_matrix(truth,prediction)
    # print('confusion_matrix:\n', cm)
    get_IoU(cm, labels, fold, branch)
    acc = np.sum(cm*np.eye(classes))/np.sum(cm)
    # print('Globale acc:',acc)
    # confusion_matrix_1 = cm/np.sum(cm,axis=0)
    # confusion_matrix_2 = cm/np.sum(cm)


    # xlocations = np.array(range(len(labels)))
    # plt.xticks(xlocations, labels,fontsize=12)
    # plt.yticks(xlocations, labels,fontsize=12)
    # plt.title(model+', Accuracy: '+'%.3f' %(acc*100)+'%', family='fantasy', fontsize=15)
    # plt.ylabel('True label', fontsize=12)
    # plt.xlabel('Predicted label', fontsize=12)
    # plt.tight_layout()
    # x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    # for x,y in zip(x.flatten(), y.flatten()):
    #     prob = confusion_matrix_1[x,y]
    #     if x==y:
    #         plt.text(x,y,"%00.2f" %(prob*100,), color='green', fontsize=10, va='center', ha='center')
    # plt.imshow(confusion_matrix_1, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.savefig(model+'_'+fold+'cm_1.png')

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
    #     # if x==y:
    #     #     plt.text(x,y,"%00.2f" %(prob*100,), color='green', fontsize=10, va='center', ha='center')
    # plt.imshow(confusion_matrix_2, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.savefig(model+'_'+fold+'cm_2.png')

def get_IoU(confusion_matrix, labels, fold, branch):
    mIoU=0
    iou_list = {}
    for i in range(len(labels)):
        iou_list[labels[i]] = confusion_matrix[i,i]/(np.sum(confusion_matrix[i,:])+np.sum(confusion_matrix[:,i])-confusion_matrix[i,i])
        mIoU += iou_list[labels[i]]
    # print(iou_list)
    print(f'{fold}\t{branch}\tmIoU:{mIoU/confusion_matrix.shape[0]}')
    # print('mIOU:', mIoU/confusion_matrix.shape[0])

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, metavar='FILE',
                        help="Specify the network name")
    parser.add_argument('--weight', '-w', type=str, metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--alpha', '-a', metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--fold', '-f', metavar='FILE', type=str,
                        help="Specify the file in which the model is stored")
    parser.add_argument('-o', '--skip', dest='skip', type=str, 
                        help='skip'),
    parser.add_argument('--gpu', '-g', metavar='FILE', type=str,
                        help="Specify the file in which the model is stored")
    parser.add_argument('--branch', '-b', dest='branch', type=int, help='branch')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.set_device(0)
    
    net = AdaBoost_UNet(n_channels=1, n_classes=28, level='1234', filters=16, skip_option=args.skip).cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Loading model {}".format(args.weight))
    logging.info(f'Using device {device}')
    # summary(net, (1, 512, 512))
    count=0
    # for name, para in net.named_parameters():
    #     print(name, para.size())
    #     size = para.size()
    #     s=1
    #     for d in size:
    #         s *= d
    #     count += s
    # print('num of paras:', count)
    net = nn.DataParallel(net, device_ids=[0])
    net.to(device=device)

    # state_dict = torch.load(args.weight, map_location=device)
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = 'module.'+k # remove `module.`
    #     new_state_dict[name] = v
    # net.load_state_dict(new_state_dict)

    net.load_state_dict(torch.load(args.weight, map_location=device))
    f=open(args.alpha,'r')
    alphas=[]
    i=0
    for line in f:
        if i==4:
            break
        alphas.append(np.float(line))
        i+=1 
    print(args.fold, args.skip, alphas)
    predict_img(args.model, net, device, args.fold, alphas,args.branch)
