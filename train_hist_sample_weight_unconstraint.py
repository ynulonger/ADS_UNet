import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from metrics import *
from dice_loss import *
from utils.loss import *
from torch import optim
from utils.dataset import *
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.ticker as ticker
from torch.autograd import Variable
from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from unet.Adaboost_UNet import AdaBoost_UNet as AdaBoost_UNet

def normalize(eta):
    eta = eta.view(-1)
    eta = F.softmax(eta, dim=0)
    return eta

def evaluate(criterion, net, loader, device, classes, level, loss_fn):
    net.eval()
    iou = 0
    loss = 0
    length = 0
    metrics = Metrics(classes)
    hidden_loss = np.zeros([1,level+1])
    for batch in loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            weight = batch['weight'].to(device=device, dtype=torch.float32)
            pool_size = [j for j in range(1, level+1)]
            pool_size.reverse()
            mask_list = [F.avg_pool2d(true_masks, kernel_size=2**i, stride=2**i) for i in pool_size]+[true_masks]
            pred_list, etas = net(imgs)
            temp_loss, temp_loss_list = compute_loss(criterion, pred_list, mask_list, net, weight, device, level, loss_fn)
            loss += temp_loss.item()
            hidden_loss += temp_loss_list
            etas = normalize(etas)
            node = torch.argmax(etas)
            preds = F.interpolate(F.softmax(pred_list[node],dim=1), size=(512,512), mode='bilinear', align_corners=True)
            preds = torch.argmax(preds, dim=1).squeeze(1)
            
            if len(true_masks.size())==4:
                masks=torch.argmax(true_masks, dim=1)
            metrics.add(preds, masks)
            
    miou = metrics.iou(average=False).cpu().numpy()
    miou = np.reshape(miou,[1,-1])
    return miou, loss, hidden_loss

def compute_loss(criterion, pred_list, mask_list, net, weight, device, level, loss_fn):
    loss = 0
    alpha_vector = []
    i=0
    loss_list = []

    loss = torch.empty(0, requires_grad=True, device=device)
    for pred, mask in zip(pred_list, mask_list):
        if loss_fn=='iou':
            temp_loss = criterion(pred, mask, weight).view(1,1)
        elif loss_fn=='ce':
            temp_loss = criterion(pred, mask).sum(dim=1)
            # print(f'temp_loss_1:{temp_loss.size()}')
            temp_loss = temp_loss.mean(dim=[1,2]).view(-1,1)
            # print(f'temp_loss_2:{temp_loss.size()}')
            # print(f'weight:{weight.size()},{weight}, loss: {temp_loss.size()}, {temp_loss}')
            temp_loss = (temp_loss*weight).sum().view(1,1)
            # print(temp_loss)

        loss_list.append(temp_loss.item())
        loss = torch.cat([loss, temp_loss], dim=1)

    if level==1:
        alpha_temp = F.softmax(net.alpha_1.view(-1), dim=0)
    elif level==2:
        alpha_temp = F.softmax(net.alpha_2.view(-1), dim=0)
    elif level==3:
        alpha_temp = F.softmax(net.alpha_3.view(-1), dim=0)
    elif level==4:
        alpha_temp = F.softmax(net.alpha_4.view(-1), dim=0)
    loss = torch.dot(loss.view(-1), alpha_temp)
    return loss, np.reshape(np.array(loss_list),[1,-1])

def train_net(net, device, train_dataset, val_dataset, epochs, batch_size, lr,level, 
            skip_option, loss_fn, dir_checkpoint, deep_sup, img_size, classes):
    class_weight = 1-torch.Tensor([0.050223350000000014, 0.40162656, 0.36744026, 0.11795166, 0.06275817]).cuda()
    class_weight = class_weight.view(1,5,1,1)
    n_train = train_dataset.__len__()
    n_val   = val_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    train_loss_list = []
    val_loss_list  = []
    train_iou_list = np.empty([0,classes])
    val_iou_list  = np.empty([0,classes])
    train_h_loss = np.empty([0,level+1])
    val_h_loss = np.empty([0,level+1])
    alphas = np.empty([0,level+1])

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        skip option      {skip_option}
    ''')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-7)
    train_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    if loss_fn=='iou':
        print('using iou loss.')
        criterion = mIoULoss(n_classes=classes).cuda()
    elif loss_fn =='ce':
        # criterion = nn.BCELoss(reduction='none').cuda()
        if 'BCSS' in dir_checkpoint:
            criterion = SoftCrossEntropy(weight=class_weight, reduction='none').cuda()
        else:
            criterion = SoftCrossEntropy(reduction='none').cuda()

    count = level+1
    best_miou=0
    for epoch in range(epochs):
        if level==1:
            alpha_vector = F.softmax(net.alpha_1.view(-1), dim=0).cpu().detach().numpy()
        elif level==2:
            alpha_vector = F.softmax(net.alpha_2.view(-1), dim=0).cpu().detach().numpy()
        elif level==3:
            alpha_vector = F.softmax(net.alpha_3.view(-1), dim=0).cpu().detach().numpy()
        elif level==4:
            alpha_vector = F.softmax(net.alpha_4.view(-1), dim=0).cpu().detach().numpy()
        print('alpha:',alpha_vector)
        alphas = np.concatenate([alphas, np.reshape(alpha_vector,[1,-1])], axis=0)

        e_start=time.time()
        # learning_rate = train_scheduler.get_last_lr()
        net.train()
        for batch in train_loader:
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            weight = batch['weight'].to(device=device, dtype=torch.float32)
            pool_size = [j for j in range(1, level+1)]
            pool_size.reverse()
            mask_list = [F.avg_pool2d(true_masks, kernel_size=2**i, stride=2**i) for i in pool_size]+[true_masks]
            pred_list, _ = net(imgs)
            optimizer.zero_grad()
            loss, _ = compute_loss(criterion, pred_list, mask_list, net, weight, device, level, loss_fn)
            # train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_scheduler.step()

        train_miou, train_loss, train_hidden_loss = evaluate(criterion, net, train_loader, device, classes, level, loss_fn)
        val_miou, val_loss, val_hidden_loss       = evaluate(criterion, net, val_loader, device, classes, level, loss_fn)

        e_end=time.time()
        print('Epoch:', epoch+1,'/',epochs, 'time:',round(e_end-e_start, 1), 'Train_Loss:', round(train_loss,4), 'Val_Loss:', round(val_loss,4), 'Train_miou:', round(train_miou.mean(),4), 'val_miou:',round(val_miou.mean(),4))
        print(f'train iou: {train_miou}, test iou: {val_miou}')
        
        train_loss_list.append(train_loss)
        train_h_loss = np.concatenate([train_h_loss, train_hidden_loss], axis=0)
        val_loss_list.append(val_loss)
        val_h_loss  = np.concatenate([val_h_loss, val_hidden_loss], axis=0)
        train_iou_list=np.concatenate([train_iou_list, train_miou], axis=0)
        val_iou_list=np.concatenate([val_iou_list, val_miou], axis=0)

        if val_miou.mean()>best_miou:
            best_miou = val_miou.mean()
            torch.save(net.state_dict(),
                    f'{dir_checkpoint}{loss_fn}_{lr}_Ada_{skip_option}_unconstraint.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    train_loss_list = np.reshape(np.array(train_loss_list),[-1,1])
    val_loss_list   = np.reshape(np.array(val_loss_list),[-1,1])
    loss_curve = np.concatenate([train_loss_list, val_loss_list], axis=1)
    iou_curve = np.concatenate([train_iou_list, val_iou_list], axis=1)
    loss_record = pd.DataFrame(loss_curve)
    iou_record = pd.DataFrame(iou_curve)
    alpha_record = pd.DataFrame(alphas)
    train_h_loss = pd.DataFrame(train_h_loss)
    val_h_loss  = pd.DataFrame(val_h_loss)
    writer = pd.ExcelWriter(f'{dir_checkpoint}{loss_fn}_{lr}_Ada_{skip_option}_{level}_unconstraint.xlsx')

    loss_record.to_excel(writer, 'loss', float_format='%.8f')       
    iou_record.to_excel(writer, 'iou', float_format='%.8f')       
    alpha_record.to_excel(writer, 'alpha', float_format='%.8f')     
    train_h_loss.to_excel(writer, 'train_hidden', float_format='%.8f')
    val_h_loss.to_excel(writer, 'test_hidden', float_format='%.8f')
    writer.save()
    writer.close()
    return net, train_hidden_loss

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=70,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('-o', '--skip', dest='skip', type=str, default='scse',
                        help='skip')
    parser.add_argument('-g', '--gpu', dest='n_gpu', type=int, default=0,
                        help='number of gpus')
    parser.add_argument('-c', '--loss', dest='loss', type=str, default='ce',
                        help='loss function')
    parser.add_argument('-d', '--data', dest='data', type=str, default='BCSS',
                        help='dataset')
    return parser.parse_args()

def get_data(train_sample_weight, dataset_name):
    if dataset_name=='BCSS':
        train_dataset = BCSS('train', sample_weight=train_sample_weight, mask_channel=5)
        test_dataset  = BCSS('test', sample_weight='sample_mean', mask_channel=5)
    elif dataset_name=='CRAG':
        train_dataset = CRAG('train', sample_weight=train_sample_weight, mask_channel=2)
        test_dataset  = CRAG('test', sample_weight='sample_mean', mask_channel=2)
    elif dataset_name=='Kumar':
        train_dataset = Kumar(image_set='train', sample_weight=train_sample_weight)
        test_dataset  = Kumar(image_set='test', sample_weight='sample_mean')
    return train_dataset, test_dataset

def eval_UNet(net, val_dataset, device, batch_size, class_num):
    net.eval()
    metrics = Metrics(class_num)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            pred_list, etas = net(imgs)
            etas = normalize(etas)
            node = torch.argmax(etas)
            preds = F.interpolate(F.softmax(pred_list[node],dim=1), size=(512,512), mode='bilinear', align_corners=True)
            preds = torch.argmax(preds, dim=1, keepdim=True)
            masks = torch.argmax(masks, dim=1, keepdim=True)
            metrics.append_iou(preds, masks)
            metrics.add(preds, masks)
    miou_list = metrics._iou_list
    miou_error= 1-miou_list
    w_error = (miou_error*val_dataset.sample_weight.view(1,-1).cuda()).sum()
    print(miou_error.size(), val_dataset.sample_weight.size(), w_error.size())
    w_miou  = (miou_list*val_dataset.sample_weight.view(1,-1).cuda()).sum()
    print('mean iou:',metrics.iou(True), 'mean error:', miou_error.mean())
    return miou_list.view(-1,1).cpu(), w_error, w_miou

if __name__ == '__main__':
    # classes = 12
    # img_size=[360, 480]
    classes = 5
    img_size= [512, 512]
    deep_sup=True
    n_channels = 3
    start = time.time()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

    print('-------------------------', args.loss)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    alphas = []

    dir_checkpoint = 'checkpoints/'+args.data+'/unconstraint/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    train_sample_weight = 'sample_mean'
    all_weight = None
    freeze=False
    for t in range(1,5):
        print(f'===========================iter:{t}==============================')
        net = AdaBoost_UNet(n_channels=n_channels, n_classes=classes, level=str(t), skip_option=args.skip, filters=64, deep_sup=deep_sup).cuda()

        if t==1:
            # net.load_state_dict(torch.load(f'{dir_checkpoint}{args.loss}_{fold}_{args.lr}_DS.pth', map_location=device))
            train_dataset, val_dataset = get_data('sample_mean', args.data)
            all_weight = np.empty([len(train_dataset.sample_weight), 0])

            for name,para in net.named_parameters():
                if name.split('.')[0] in ['X_00', 'X_10', 'up_10', 'out_01', 'out_10', 'alpha_1']:
                    para.requires_grad=True
                else:
                    para.requires_grad=False

        else:
            train_dataset, val_dataset = get_data(train_sample_weight, args.data)            
            if os.path.exists(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}_unconstraint.pth'):
                print(f'loading weight from {dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}_unconstraint.pth')
                net.load_state_dict(torch.load(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}_unconstraint.pth', map_location=device))
            else:
                print('train from initialization.')

            depth=t+1
            
            if freeze:
                trainable_paras = []
                for i in range(depth):
                    trainable_paras.append('out_'+str(i)+str(t-i))
                idx_1 = [i for i in range(t)]
                idx_2 = [i for i in range(1, depth)]
                idx_2.reverse()
                for i,j in zip(idx_2, idx_1):
                    trainable_paras.append('up_'+str(i)+str(j))
                trainable_paras.append('X_'+str(t)+'0')
                trainable_paras.append('alpha_'+str(t))

                for name,para in net.named_parameters():
                    if name.split('.')[0] in trainable_paras:
                        para.requires_grad=True
                    else:
                        para.requires_grad=False

            else:
                encoder_nodes = ['X_'+str(i)+'0' for i in range(depth)]
                decoder_nodes = ['up_'+str(i)+str(t-i) for i in range(1, depth)]
                out_layer_nodes=['out_'+str(i)+str(t-i) for i in range(depth)]
                for name,para in net.named_parameters():
                    if name[:5] in decoder_nodes:
                        para.requires_grad=True
                    elif name[:6] in out_layer_nodes:
                        para.requires_grad=True
                    elif name[:4] in encoder_nodes:
                        para.requires_grad=True
                    elif 'alpha' in name:
                        para.requires_grad=True
                    else:
                        para.requires_grad=False

        train_sample_weight = train_dataset.sample_weight
        all_weight = np.concatenate([all_weight, np.reshape(train_sample_weight.cpu().numpy(),[-1,1])], axis=1)

        summary(net, (n_channels, img_size[0], img_size[1]))
        net.to(device=device)

        net, train_hidden_loss = train_net(net=net, device=device, train_dataset= train_dataset, val_dataset=val_dataset,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,level=t, 
                skip_option=args.skip, loss_fn=args.loss,dir_checkpoint=dir_checkpoint, 
                deep_sup=deep_sup, img_size=img_size, classes=classes)

        print(f'loading weight from {dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}_unconstraint.pth')
        net.load_state_dict(torch.load(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}_unconstraint.pth', map_location=device))
        
        head = np.argmax(train_hidden_loss)
        start_e = time.time()
        mIoU_list, w_err, w_miou = eval_UNet(net, train_dataset, device, args.batch_size, classes)
        end_e = time.time()
        print(f'evaluation time: {round(end_e-start_e, 2)}, IoU:{round(mIoU_list.mean().item(),3)}, w_IoU:{round(w_miou.item(),3)}, w_err:{round(w_err.item(),3)}, sample_weight:{train_sample_weight.view(-1)}')
        if w_err<(1-1/classes):
            freeze = True
            alpha=torch.log((1-w_err)/w_err)/2+torch.log(torch.tensor(classes-1, dtype=torch.float32))
            train_sample_weight = train_sample_weight*torch.exp(1-mIoU_list)
            train_sample_weight = train_sample_weight/torch.sum(train_sample_weight)
        else:
            freeze = False
            alpha=torch.tensor(0)
        print('alpha_'+str(t)+':', alpha.item(), 'head:', head)
        alphas.append(str(head)+' '+str(alpha.item()))

    end = time.time()
    print('alphas:',alphas, 'running time:', end-start)
    f = open(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}_unconstraint.txt', 'w')

    for alpha in alphas:
        f.write(str(alpha))
        f.write("\n")
    # f.write('running time:'+str(end-start))
    f.close()

    all_weight = pd.DataFrame(all_weight)
    writer = pd.ExcelWriter(f'{dir_checkpoint}{args.loss}_{deep_sup}_{args.lr}_unconstraint.xlsx')

    all_weight.to_excel(writer, 'weight', float_format='%.8f')
    writer.save()
    writer.close()


