import os
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from metrics import *
from dice_loss import *
from torch import optim
# from thop import profile
from utils.loss import *
from utils.dataset import *
# from thop import clever_format
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.ticker as ticker
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from unet.deep_supervise_unet_model import DS_CNN
from torch.utils.data import DataLoader, random_split

def evaluate(criterion, net, loader, device, classes, loss_fn):
    net.eval()
    iou = 0
    loss = 0
    length = 0
    metrics = Metrics(classes)
    hidden_loss = np.zeros([1,8])
    for batch in loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            pred_list = net(imgs)
            temp_loss, temp_loss_list = compute_loss(criterion, pred_list, true_masks, net, device, loss_fn)
            loss += temp_loss.item()
            hidden_loss += temp_loss_list
            preds = F.softmax(pred_list[-1],dim=1)

            masks = torch.argmax(true_masks, dim=1).squeeze(1)
            preds = torch.argmax(preds, dim=1).squeeze(1)
            metrics.add(preds, masks)
            length += 1

    miou = metrics.iou(average=False).cpu().numpy()
    miou = np.reshape(miou,[1,-1])
    return miou, loss/length, hidden_loss/length

def compute_loss(criterion, pred_list, mask, net, device, loss_fn):
    loss = 0
    alpha_vector = []
    i=0
    loss_list = []
    loss = torch.empty(0, requires_grad=True, device =device)
    for pred in pred_list:
        # print(pred.size(), mask.size() )
        temp_loss = criterion(pred,  torch.argmax(mask,dim=1)).view(1,1)
        loss_list.append(temp_loss.item())
        loss = torch.cat([loss, temp_loss], dim=1)

    alpha_temp = F.softmax(net.alpha.view(-1), dim=0)
    loss = torch.dot(loss.view(-1), alpha_temp)

    # loss = loss.mean()
    return loss, np.array(loss_list)

def train_net(net, device, train_dataset, val_dataset, epochs, batch_size, 
                lr, loss_fn, dir_checkpoint, img_size, model, classes, seed, data_name):
    class_weight = 1-torch.Tensor([0.050223350000000014, 0.40162656, 0.36744026, 0.11795166, 0.06275817]).cuda()
    n_train = train_dataset.__len__()
    n_val   = val_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    train_loss_list = []
    val_loss_list  = []
    train_h_loss = np.empty([0,8])
    val_h_loss = np.empty([0,8])
    train_iou_list = np.empty([0,classes])
    val_iou_list  = np.empty([0,classes])
    # epslion = 1/8

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Dataset:         {data_name}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Loss function:   {loss_fn}
        model name:      {model}
    ''')

    alphas = np.empty([0,8])
    alpha_vector = F.softmax(net.alpha.view(-1), dim=0).view(1,8).cpu().detach().numpy()
    alphas = np.concatenate([alphas, alpha_vector], axis=0)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-7)
    train_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    if loss_fn =='ce' or loss_fn=='CE':
        if 'BCSS' in dir_checkpoint:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean').cuda()
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='mean').cuda()

    elif loss_fn=='iou' or loss_fn=='dice':
        print('using iou loss')
        criterion = mIoULoss(n_classes=classes).cuda()

    best_miou=0
    for epoch in range(epochs):
        e_start=time.time()
        net.train()
        train_loss = 0
        for batch in train_loader:
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            pred_list = net(imgs)
            optimizer.zero_grad()
            loss, loss_list = compute_loss(criterion, pred_list, true_masks, net, device, loss_fn)
            loss.backward()
            optimizer.step()
            train_scheduler.step()

        train_miou, train_loss, train_hidden_loss = evaluate(criterion, net, train_loader, device, classes, loss_fn)
        val_miou, val_loss, val_hidden_loss = evaluate(criterion, net, val_loader, device, classes, loss_fn)

        e_end=time.time()
        # print(f'Epoch:{epoch+1}/{epochs}, time:{round(e_end-e_start, 1)}, Loss:, {round(train_loss,4)},{round(val_loss,4)}, IoU:, {train_miou.mean()}, {val_miou.mean()}')
        if data_name=='CRAG' or data_name=='Kumar':
            print(f"Epoch:{epoch+1}/{epochs}, time:{round(e_end-e_start, 1)}, Train_Loss:, {format(train_loss,'.4f')}, Test_Loss: {format(val_loss, '.4f')}, Train_mIoU:, {format(train_miou[0,1],'.4f')}, Test_mIoU: {format(val_miou[0,1],'.4f')}")
        elif data_name=='BCSS':
            print(f"Epoch:{epoch+1}/{epochs}, time:{round(e_end-e_start, 1)}, Train_Loss:, {format(train_loss,'.4f')}, Test_Loss: {format(val_loss, '.4f')}, Train_mIoU:, {format(train_miou.mean(),'.4f')}, Test_mIoU: {format(val_miou.mean(),'.4f')}")

        train_h_loss = np.concatenate([train_h_loss, train_hidden_loss], axis=0)
        train_loss_list.append(train_loss)
        val_h_loss  = np.concatenate([val_h_loss, val_hidden_loss], axis=0)
        val_loss_list.append(val_loss)

        train_iou_list=np.concatenate([train_iou_list, train_miou], axis=0)
        val_iou_list=np.concatenate([val_iou_list, val_miou], axis=0)

        alpha_vector=F.softmax(net.alpha.view(-1), dim=0).view(1,8).cpu().detach().numpy()
        alphas = np.concatenate([alphas, alpha_vector], axis=0)

        if val_miou.mean()>best_miou:
            best_miou = val_miou.mean()
            torch.save(net.state_dict(), dir_checkpoint + f'{seed}_{lr}_DS_{loss_fn}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    train_iou_list = np.reshape(np.array(train_iou_list),[-1,1])
    val_iou_list = np.reshape(np.array(val_iou_list),[-1,1])

    train_loss_list = np.reshape(np.array(train_loss_list),[-1,1])
    val_loss_list   = np.reshape(np.array(val_loss_list),[-1,1])
    loss_curve = np.concatenate([train_loss_list, val_loss_list], axis=1)
    iou_curve = np.concatenate([train_iou_list, val_iou_list], axis=1)
    loss_record = pd.DataFrame(loss_curve)
    iou_record = pd.DataFrame(iou_curve)
    alpha_record = pd.DataFrame(alphas)

    train_h_loss = pd.DataFrame(train_h_loss)
    val_h_loss  = pd.DataFrame(val_h_loss)

    writer = pd.ExcelWriter(f'{dir_checkpoint}/{seed}_{lr}_DCNN_{loss_fn}.xlsx')      

    loss_record.to_excel(writer, 'loss', float_format='%.8f')       
    iou_record.to_excel(writer, 'iou', float_format='%.8f')       
    train_h_loss.to_excel(writer, 'train_hidden', float_format='%.8f')
    val_h_loss.to_excel(writer, 'test_hidden', float_format='%.8f')
    alpha_record.to_excel(writer, 'alpha', float_format='%.8f')     
    writer.save()

    writer.close()
    return net

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=70,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-w', '--weight', dest='weight', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, 
                        help='batch size', default=4)
    parser.add_argument('-o', '--loss', dest='loss', type=str, default='ce',
                        help='loss function')
    parser.add_argument('-g', '--gpu', dest='n_gpu', type=int, default=0,
                        help='number of gpus')
    parser.add_argument('-d', '--data', dest='data', type=str,  default='BCSS',
                        help='dataset')
    parser.add_argument('-m', '--model', dest='model', type=str, default='DCNN',
                        help='model')
    parser.add_argument('-s', '--seed', dest='seed', type=str, default=34, 
                        help='mask')
    return parser.parse_args()

def count_your_model(model, x, y):
    # your rule here
    pass
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    dataset = args.data

    dir_checkpoint = 'checkpoints/'+dataset+'/'+args.model+'/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    if args.data == 'BCSS':
        classes =5
        train_dataset = BCSS('train', sample_weight=None, mask_channel=5)
        test_dataset  = BCSS('test', sample_weight=None, mask_channel=5)
        img_size= [512, 512]

    elif args.data=='CRAG':
        classes =2
        train_dataset = CRAG('train', mask_channel=2)
        test_dataset  = CRAG('test',  mask_channel=2)
        img_size= [512, 512]
    elif args.data=='Kumar':
        classes =2
        train_dataset = Kumar(image_set='train', mask_channel=2)
        test_dataset  = Kumar(image_set='test', mask_channel=2)
        img_size= [400,400]
    
    net = DS_CNN(n_channels=3, n_classes=classes, filters=64, bilinear=False).cuda()
    summary(net, (3, img_size[0], img_size[1]))
    net.to(device=device)

    # inputs = torch.randn(1, 3, 512, 512).to(device=device)
    # flops, params = profile(net, inputs=(inputs, ), custom_ops={DS_CNN: count_your_model})
    # flops, params = clever_format([flops, params], "%.3f")
    # print("flops", flops, "params", params)

    if args.weight:
        net.load_state_dict(torch.load(args.weight, map_location=device))
        logging.info(f'Model loaded from {args.weight}')

    net = train_net(net=net, device=device, train_dataset= train_dataset, val_dataset=test_dataset,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, loss_fn=args.loss, 
                    dir_checkpoint=dir_checkpoint, img_size=img_size, model=args.model, classes=classes, 
                    seed=args.seed, data_name = dataset)
