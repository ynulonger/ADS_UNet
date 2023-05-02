import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from metrics import *
from unet import UNet
from tqdm import tqdm
from dice_loss import *
from torch import optim
from utils.loss import *
from utils.dataset import *
import torch.nn.functional as F
from torchsummary import summary
from unet.unet_model import CENet
from unet.unet_pp import UNet2Plus
import matplotlib.ticker as ticker
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

def evaluate(criterion, net, loader, device, classes):
    net.eval()
    metrics = Metrics(classes)
    iou = 0
    loss = 0
    length = 0

    for batch in loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.long))
            masks_pred = net(imgs)
            temp_loss = criterion(masks_pred, torch.argmax(true_masks, dim=1))
            loss += temp_loss.item()
            preds = F.softmax(masks_pred,dim=1)
            preds = torch.argmax(preds, dim=1).squeeze(1)
            if len(true_masks.size())==4:
                true_masks=torch.argmax(true_masks, dim=1)
            masks = true_masks.view(-1)
            preds = preds.view(-1)
            metrics.add(preds, masks)
            length += 1
    miou = metrics.iou(average=False).cpu().numpy()
    miou = np.reshape(miou,[1,-1])
    return miou, loss/length

def train_net(net, device, train_dataset, val_dataset, epochs, batch_size, lr, model, 
                dir_checkpoint, loss_fn, img_size, classes, fold):
    class_weight = 1-torch.Tensor([0.050223350000000014, 0.40162656, 0.36744026, 0.11795166, 0.06275817]).cuda()
    n_train = train_dataset.__len__()
    n_val   = val_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    train_loss_list = []
    val_loss_list  = []

    train_iou_list = np.empty([0,classes])
    val_iou_list  = np.empty([0,classes])

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        model name:      {model}
        Fold:            {fold}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    if loss_fn=='iou':
        print('using iou loss')
        criterion = mIoULoss(n_classes=classes).cuda()
    else:
        print('using CE loss')
        # criterion = SoftCrossEntropy(reduction='mean').cuda()
        if 'BCSS' in dir_checkpoint:
            criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='mean').cuda()
        else:
            criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    train_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    best_miou=0
    for epoch in range(epochs):
        e_start=time.time()
        net.train()
        train_loss = 0
        for batch in train_loader:
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.long))
            masks_pred = net(imgs)
            optimizer.zero_grad()
            # print('mask', torch.unique(torch.argmax(true_masks, dim=1)))
            loss = criterion(masks_pred, torch.argmax(true_masks, dim=1))

            train_loss += loss
            loss.backward()
            optimizer.step()
            train_scheduler.step()

        train_miou, train_loss = evaluate(criterion, net, train_loader, device, classes)
        val_miou, val_loss = evaluate(criterion, net, val_loader, device, classes)
        
        e_end=time.time()
        print(f'Epoch:{epoch+1}/{epochs}, time:{round(e_end-e_start, 1)}, Train_Loss:, {round(train_loss,4)}, Test_Loss: {round(val_loss,4)}, Train_mIoU:, {round(train_miou.mean(),4)}, Test_mIoU: {round(val_miou.mean(),4)}')
        print(f'train iou: {train_miou}, test iou: {val_miou}')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_iou_list=np.concatenate([train_iou_list, train_miou], axis=0)
        val_iou_list=np.concatenate([val_iou_list, val_miou], axis=0)

        if val_miou.mean()>best_miou:
            best_miou = val_miou.mean()
            torch.save(net.state_dict(), dir_checkpoint + f'{fold}_{batch_size}_{lr}_{model}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')  
            
    train_loss_list = np.reshape(np.array(train_loss_list),[-1,1])
    val_loss_list   = np.reshape(np.array(val_loss_list),[-1,1])


    loss_curve = np.concatenate([train_loss_list, val_loss_list], axis=1)
    iou_curve = np.concatenate([train_iou_list, val_iou_list], axis=1)

    loss_record = pd.DataFrame(loss_curve)
    iou_record = pd.DataFrame(iou_curve)

    writer = pd.ExcelWriter(dir_checkpoint+f'{fold}_{batch_size}_{lr}_{model}.xlsx')      
    loss_record.to_excel(writer, 'loss', float_format='%.8f')       
    iou_record.to_excel(writer, 'iou', float_format='%.8f')
    writer.save()

    writer.close()
    return net

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=70,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('-m', '--model', dest='model', type=str, default='UNet',
                        help='model')
    parser.add_argument('-g', '--gpu', dest='n_gpu', type=int, default=0,
                        help='number of gpus')
    parser.add_argument('-d', '--data', dest='data', type=str, default='BCSS',
                        help='dataset')
    parser.add_argument('-c', '--cost', dest='cost', type=str, default='ce',
                        help='dataset')
    parser.add_argument('-f', '--fold', dest='fold', type=int, default=0, 
                        help='mask')
    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    dataset = args.data
    dir_checkpoint = 'checkpoints/'+dataset+'/'+args.model+'/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    if args.data == 'BCSS':
        classes =5
        img_size= [512,512]
        train_dataset = BCSS('train', sample_weight=None, mask_channel=5, fold=args.fold)
        test_dataset  = BCSS('test', sample_weight=None, mask_channel=5, fold=args.fold)
    elif args.data=='CRAG':
        classes =2
        img_size= [512,512]
        train_dataset = CRAG('train', mask_channel=2, fold=args.fold)
        test_dataset  = CRAG('test',  mask_channel=2, fold=args.fold)
    elif args.data=='Kumar':
        classes =2
        img_size= [400,400]
        train_dataset = Kumar('train', fold=args.fold)
        test_dataset  = Kumar('test', fold=args.fold)

    if args.model =='UNet':
        net = UNet(n_channels=3, n_classes=classes, filters=64, bilinear=False).cuda()
    elif args.model =='CENet':
        net = CENet(n_channels=3, n_classes=classes, filters=64, bilinear=False).cuda()

    summary(net, (3, img_size[0], img_size[1]))
    net.to(device=device)

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    net = train_net(net=net, device=device, train_dataset= train_dataset, val_dataset=test_dataset,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model =args.model,
                    dir_checkpoint=dir_checkpoint, loss_fn=args.cost, img_size=img_size, 
                    classes=classes, fold=args.fold)
    # predict_img(net, val_dataset, device)
