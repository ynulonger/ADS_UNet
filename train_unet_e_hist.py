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
from tqdm import tqdm
from dice_loss import *
from torch import optim
from utils.dataset import *
from unet.unet_e import UNet_e
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.ticker as tickers
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

def evaluate(criterion, net, loader, device, classes):
    net.eval()
    iou = 0
    loss = 0
    length = 0
    metrics = Metrics(classes)
    for batch in loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            masks_pred = net(imgs)
            # temp_loss = criterion(masks_pred, torch.argmax(true_masks, dim=1))
            preds=0
            for output in masks_pred:
                loss += criterion(output, torch.argmax(true_masks, dim=1)).item()
                preds += F.softmax(output,dim=1)
            # masks = torch.argmax(true_masks, dim=1).squeeze(1)
            preds = torch.argmax(preds, dim=1)
            if len(true_masks.size())==4:
                true_masks=torch.argmax(true_masks, dim=1)
            masks = true_masks.view(-1)
            preds = preds.view(-1)
            metrics.add(preds, masks)
            length += 1
    iou = metrics.iou(average=False).view(1,-1).cpu().numpy()
    return iou, loss/(length*4)

def train_net(net, device, train_dataset, val_dataset, epochs, batch_size, 
            lr, model, dir_checkpoint, img_size, classes, loss_fn, seed):
    class_weight = 1-torch.Tensor([0.050223350000000014, 0.40162656, 0.36744026, 0.11795166, 0.06275817]).cuda()
    n_train = train_dataset.__len__()
    n_val   = val_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    train_loss_list = []
    val_loss_list  = []
    train_iou_list = np.empty([0,classes])
    val_iou_list = np.empty([0,classes])

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        model name:      {model}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)
    train_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    # criterion = nn.BCELoss(reduction='mean').cuda()
    if loss_fn =='iou':
        print('using iou loss')
        criterion = mIoULoss(n_classes=classes).cuda()
    elif loss_fn=='bce':
        print('using bce loss')
        criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()
    elif loss_fn =='ce' or loss_fn=='CE':
        print('using ce loss')
        if 'BCSS' in dir_checkpoint:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean').cuda()
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    best_miou=0
    for epoch in range(epochs):
        e_start=time.time()
        # learning_rate = train_scheduler.get_last_lr()
        net.train()
        for batch in train_loader:
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            outputs = net(imgs)
            optimizer.zero_grad()
            loss=0
            for output in outputs:
                loss += criterion(output, torch.argmax(true_masks, dim=1))
            loss.backward()
            optimizer.step()
            train_scheduler.step()

        train_miou, train_loss = evaluate(criterion, net, train_loader, device, classes)
        val_miou, val_loss = evaluate(criterion, net, val_loader, device, classes)
        
        e_end=time.time()
        print(f"Epoch:{epoch+1}/{epochs}, time:{round(e_end-e_start, 1)}, Train_Loss:, {format(train_loss,'.4f')}, Test_Loss: {format(val_loss, '.4f')}, Train_mIoU:, {format(train_miou[0,1],'.4f')}, Test_mIoU: {format(val_miou[0,1],'.4f')}")
        print(train_miou, val_miou)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_iou_list = np.concatenate([train_iou_list, train_miou], axis=0)
        val_iou_list = np.concatenate([val_iou_list, val_miou], axis=0)

        if val_miou.mean()>best_miou:
            best_miou = val_miou.mean()
            torch.save(net.state_dict(), f'{dir_checkpoint}{seed}_{lr}_{model}_{loss_fn}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    train_loss_list = np.reshape(np.array(train_loss_list),[-1,1])
    val_loss_list   = np.reshape(np.array(val_loss_list),[-1,1])
    loss_curve = np.concatenate([train_loss_list, val_loss_list], axis=1)
    loss_record = pd.DataFrame(loss_curve)
    train_iou_record = pd.DataFrame(train_iou_list)
    val_iou_record = pd.DataFrame(val_iou_list)
    writer = pd.ExcelWriter(f'{dir_checkpoint}{seed}_{lr}_{loss_fn}_{model}_.xlsx')      
    loss_record.to_excel(writer, 'loss', float_format='%.8f')       
    train_iou_record.to_excel(writer, 'train_iou', float_format='%.8f')       
    val_iou_record.to_excel(writer, 'val_iou', float_format='%.8f')       
    writer.save()
    writer.close()
    return net

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=70,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('--gpu', '-g', metavar='FILE', type=str, default='0',
                        help="Specify the file in which the model is stored")
    parser.add_argument('-d', '--data', dest='data', type=str, default='BCSS',
                        help='dataset')
    parser.add_argument('-c', '--cost', dest='cost', type=str, default='ce',
                        help='dataset')
    parser.add_argument('-m', '--model', dest='model', type=str, default='UNet_e',
                        help='model')
    parser.add_argument('-s', '--seed', dest='seed', type=str, default=34, 
                        help='mask')
    return parser.parse_args()

if __name__ == '__main__':
    classes = 5
    img_size= [512,512]

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    dataset = args.data
    dir_checkpoint = 'checkpoints/'+dataset+'/'+args.model+'/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    if args.data == 'BCSS':
        classes = 5
        train_dataset = BCSS('train', sample_weight=None, mask_channel=5)
        test_dataset  = BCSS('test', sample_weight=None, mask_channel=5)
    elif args.data=='CRAG':
        classes = 2
        train_dataset = CRAG('train', mask_channel=2)
        test_dataset  = CRAG('test',  mask_channel=2)
    elif args.data=='Kumar':
        classes =2
        img_size= [400,400]
        train_dataset = Kumar('train')
        test_dataset  = Kumar('test')
        
    net = UNet_e(n_channels=3, n_classes=classes, filters=64).cuda()
    summary(net, (3, img_size[0], img_size[1]))
    net.to(device=device)

    net = train_net(net=net, device=device, train_dataset= train_dataset, val_dataset=test_dataset,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, model =args.model,
                    dir_checkpoint=dir_checkpoint, img_size=img_size, classes=classes, 
                    loss_fn=args.cost, seed=args.seed)
