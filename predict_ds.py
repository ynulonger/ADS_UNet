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
from unet.unet_e import UNet_e
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from unet.unet_pp import UNet2Plus
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from unet.noskip_unet import UNet as noskip_unet
from unet.deep_supervise_unet_model import DS_UNet_1 as DS_UNet
from unet.deep_supervise_unet_model import DS_CNN, DS_UNet_deeper



def predict_img(model, net, device,classes, dataset):
    save_patches = 'data/'+dataset+'/'+model+'/'
    if not os.path.exists(save_patches):
        os.makedirs(save_patches)
    net.eval()

    if dataset=='BCSS':
        test_dataset  = BCSS(image_set='test')
        img_size = (512,512)

    elif dataset=='CRAG':
        test_dataset  = CRAG(image_set='test',  mask_channel=2)
        img_size = (512,512)
    elif dataset=='Kumar':
        test_dataset  = Kumar(image_set='test',  mask_channel=2)
        img_size = (992, 992)

    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    IMG_Metrics=Metrics(classes)
    metrics = Metrics(classes)
    count=0
    miou_averaged = 0
    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            names = batch['name']
            # if model=='UNet++':
            #     output = F.softmax(net(imgs)[-1], dim=1)
            if model=='DS_UNet' or model=='DS_UNet_deeper' or model=='DS_CNN':
                # etas = F.softmax(net.alpha.view(-1).squeeze(), dim=0)
                # print(f'etas: {etas}')
                outputs = net(imgs)
                output=0
                # for i in range(0,len(etas)):
                #     output += F.interpolate(F.softmax(outputs[i],dim=1), size=(mask.size(-2), mask.size(-1)), mode='bilinear', align_corners=True)*etas[i]
                i=-1
                output += F.interpolate(F.softmax(outputs[i],dim=1), size=img_size, mode='bilinear', align_corners=True)
                
            elif model == 'UNet_e' or model=='UNet++':
                outputs = net(imgs)
                output = (F.softmax(outputs[0], dim=1)+F.softmax(outputs[1], dim=1)+F.softmax(outputs[2], dim=1)+F.softmax(outputs[3], dim=1))/4.0
            else:
                output = F.softmax(net(imgs), dim=1)
            # print('img size:', imgs.size())
            # print('output size:', output.size())

            torch_masks = torch.argmax(masks,dim=1)
            torch_preds = torch.argmax(output,dim=1)
            metrics.add(torch_preds.squeeze(1).view(-1), torch_masks.squeeze(1).view(-1))

            output = output.cpu().numpy()
            masks = masks.cpu().numpy()
            masks = np.argmax(masks, axis=1).squeeze()
            preds = np.argmax(output, axis=1).squeeze()

            if len(masks.shape)==2: 
                masks=np.expand_dims(masks,axis=0)
                preds=np.expand_dims(preds,axis=0)

            for i in range(preds.shape[0]):
                img_iou = metrics.get_img_iou(torch_preds[i], torch_masks[i]).item()
                print(names[i].split('.')[0], img_iou)
                # heatmap = (preds[i]*(255/4)).astype(np.uint8)
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
                # cv2.imwrite(save_patches+'/'+names[i].split('.')[0]+'_'+str(img_iou)[:5]+'.png', heatmap)
                count+=1
                # cv2.imwrite(save_patches+'/m_'+names[i], preds[i])

    miou_list = metrics._iou_list
    cm = metrics._confusion_matrix
    print(f'iou: {metrics.iou(average=True)}, dice:{metrics.dice(average=True)}, {metrics.iou(average=False)}')
    # miou = metrics.iou(average=False)
    # print(f'mean: {miou.mean()}, {miou}')

def get_confusion_matrix(prediction, truth,  classes, model='DS_UNet'):
    labels = [i for i in range(classes-1)]

    cm = confusion_matrix(truth,prediction)
    # print('confusion_matrix:\n', cm)
    get_IoU(cm, labels)
    acc = np.sum(cm*np.eye(classes))/np.sum(cm)
    print('Globale acc:',acc)
    confusion_matrix_1 = cm/np.sum(cm,axis=0)
    # confusion_matrix_2 = cm/np.sum(cm)


    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels,fontsize=12)
    plt.yticks(xlocations, labels,fontsize=12)
    plt.title(model+', Accuracy: '+'%.3f' %(acc*100)+'%', family='fantasy', fontsize=15)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    for x,y in zip(x.flatten(), y.flatten()):
        prob = confusion_matrix_1[x,y]
        if x==y:
            plt.text(x,y,"%00.2f" %(prob*100,), color='green', fontsize=10, va='center', ha='center')
    plt.imshow(confusion_matrix_1, cmap=plt.cm.jet)
    plt.colorbar()
    plt.savefig(model+'_cm_1.png')

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
    # plt.savefig(model+'_cm_2.png')

def get_IoU(confusion_matrix, labels):
    mIoU=0
    iou_list = {}
    for i in range(len(labels)):
        iou_list[labels[i]] = confusion_matrix[i,i]/(np.sum(confusion_matrix[i,:])+np.sum(confusion_matrix[:,i])-confusion_matrix[i,i])
        mIoU += iou_list[labels[i]]
    print(iou_list)
    print('mIOU:', mIoU/confusion_matrix.shape[0])

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', type=str, metavar='FILE',
                        help="Specify the network name")
    parser.add_argument('--weight', '-w', type=str, metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-d', '--data', dest='data', type=str, default='BCSS',
                        help='dataset')
    return parser.parse_args()

def count_your_model(model, x, y):
        # your rule here
        pass

if __name__ == "__main__":
    filters=64
    n_channels=3
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = get_args()
    torch.cuda.set_device(0)
    if args.data=='BCSS':
        classes=5
    else:
        classes=2
    
    if args.model == 'DS_UNet':
        net = DS_UNet(n_channels=n_channels, n_classes=classes, filters=filters, bilinear=False).cuda()
    elif args.model == 'DS_UNet_deeper':
        net = DS_UNet_deeper(n_channels=n_channels, n_classes=classes, filters=16, bilinear=False).cuda()
    elif args.model =='UNet':
        net = UNet(n_channels=n_channels, n_classes=classes, filters=filters, bilinear=False).cuda()
    elif args.model =='DS_CNN':
        net = DS_CNN(n_channels=n_channels, n_classes=classes, filters=filters, bilinear=False).cuda()
    elif args.model =='UNet++':
        print('using unet++')
        net = UNet2Plus(n_channels=n_channels, n_classes=classes, filters=filters).cuda()
    elif args.model =='UNet_e':
        print('using UNet_e')
        net = UNet_e(n_channels=n_channels,n_classes=classes, filters=filters).cuda()

    print(f'---------------Model:{args.model}, Dataset:{args.data}---------------')    
    logging.info("Loading model {}".format(args.weight))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.weight, map_location=device))
    # summary(net, (3, 512, 512))

    if args.model=='DS_UNet':
        # epslion=1/8.0
        etas = F.softmax(net.alpha.view(-1).squeeze(), dim=0)
        # etas = etas/etas.sum()
        print(etas)

    predict_img(args.model, net, device, classes, args.data)
