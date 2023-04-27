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
from unet.boost_unet import AdaBoost_UNet
# from unet.Adaboost_UNet_1 import AdaBoost_UNet_1 as AdaBoost_UNet

def predict_img(model, net, device, dataset, alphas, branch, test_fold=None, vote=None):
    save_patches = 'data/'+dataset+'/'+model+'/'
    if not os.path.exists(save_patches):
        os.makedirs(save_patches)

    net.eval()
    # np.random.seed(2)

    folders = ['fold_'+str(i) for i in range(5)]
    img_pathes = []
    mask_pathes = []

    for fold in folders:
        temp_imgs = glob.glob('/ssd/kylberg/'+fold+'/imgs/*png')
        temp_imgs.sort()
        img_pathes += temp_imgs
        
        temp_masks = glob.glob('/ssd/kylberg/'+fold+'/gt/*png')
        temp_masks.sort()
        mask_pathes += temp_masks

    img_pathes = np.array(img_pathes)
    mask_pathes= np.array(mask_pathes)
    fold_size=len(img_pathes)//5
    idx_list = [i for i in range(len(img_pathes))]

    test_idx_list   = [i for i in range(test_fold*fold_size, (test_fold+1)*fold_size)]
    test_img_pathes =img_pathes[np.array(test_idx_list)]
    test_mask_pathes=mask_pathes[np.array(test_idx_list)]
    test_dataset   = CustomImageSet(test_img_pathes, test_mask_pathes, sample_weight=None, mask_channel=28)
    val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
 
    metrics = Metrics(28)
    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            names = batch['name']

            output = net(imgs)
            preds = 0

            if branch<14:
                # preds = F.softmax(output[branch],dim=1)
                batch_prob = F.softmax(output[branch],dim=1)
                if batch_prob.size()[2] != 512:
                    batch_prob = F.interpolate(batch_prob, size=(512,512), mode='bilinear', align_corners=True)
                preds = batch_prob
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
                        batch_prob = F.interpolate(batch_prob, size=(512,512), mode='bilinear', align_corners=True)
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

            torch_masks = torch.argmax(masks,dim=1)
            torch_preds = torch.argmax(preds,dim=1)
            metrics.append_iou(torch_preds, torch_masks)

            preds = preds.cpu().numpy()
            masks = np.argmax(masks.cpu().numpy(), axis=1).squeeze()
            preds = np.argmax(preds, axis=1).squeeze()

            if len(masks.shape)==2: 
                masks=np.expand_dims(masks,axis=0)
                preds=np.expand_dims(preds,axis=0)

            # for i in range(preds.shape[0]):
            #     heatmap = (preds[i]*(256/27)).astype(np.uint8)
            #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            #     cv2.imwrite(save_patches+'/'+names[i], heatmap)
            #     cv2.imwrite(save_patches+'/m_'+names[i], preds[i])

    # miou = metrics.iou(average=False)
    # print(f'fold_{test_fold}, branch:{branch}, vote:{vote}, mean: {miou.mean()}')
    # print(miou)
    miou_list = metrics._iou_list
    print(f'fold_{test_fold}, branch:{branch}, vote:{vote}, mean: {miou_list.mean()}')
    # print(miou_list)

def get_confusion_matrix(prediction, truth, model, fold, branch):
    classes=12
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

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, metavar='FILE',
                        help="Specify the network name")
    parser.add_argument('--weight', '-w', type=str, metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--alpha', '-a', metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--fold', '-f', metavar='FILE', type=int, default=None,
                        help="Specify the file in which the model is stored")
    parser.add_argument('--gpu', '-g', metavar='FILE', type=str, default='0',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--branch', '-b', dest='branch', type=int, help='branch')
    parser.add_argument('--vote', '-v', dest='vote', type=str, help='branch')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.set_device(0)
    net = AdaBoost_UNet(n_channels=1, n_classes=28, iteration=1234, filters=16, skip_option='scse').cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Loading model {}".format(args.weight))
    logging.info(f'Using device {device}')
    # summary(net, (1, 512, 512))
    # count=0
    # for name, para in net.named_parameters():
    #     print(name, para.size())
    #     size = para.size()
    #     s=1
    #     for d in size:
    #         s *= d
    #     count += s
    # print('num of paras:', count)
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
        # if i==4:
        #     break
        alphas.append(np.float(line))
        # i+=1 
    print(alphas, args.vote)
    predict_img(args.model, net, device, 'kylberg', alphas, args.branch, args.fold, vote=args.vote)
