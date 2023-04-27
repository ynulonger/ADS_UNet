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
from torch import optim
from utils.loss import *
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
    epslion = 1/len(eta)
    eta = F.softmax(eta, dim=0)+epslion
    eta = eta/eta.sum()
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
            weight_list = [F.avg_pool2d(weight, kernel_size=2**i, stride=2**i)*((2**i)**2) for i in pool_size]+[weight]
            pred_list, etas = net(imgs)
            temp_loss, temp_loss_list = compute_loss(criterion, pred_list, mask_list, net, weight_list, device, level, loss_fn)
            loss += temp_loss.item()
            hidden_loss += temp_loss_list

            etas = normalize(etas)
            preds=0
            for i in range(0,len(etas)):
                preds += F.interpolate(F.softmax(pred_list[i],dim=1), size=(512,512), mode='bilinear', align_corners=True)*etas[i]
            preds = torch.argmax(preds, dim=1).squeeze(1)
            
            if len(true_masks.size())==4:
                masks=torch.argmax(true_masks, dim=1)
            metrics.add(preds, masks)
            
    miou = metrics.iou(average=False).cpu().numpy()
    miou = np.reshape(miou,[1,-1])
    return miou, loss, hidden_loss

def compute_loss(criterion, pred_list, mask_list, net, weight_list, device, level, loss_fn):
    loss = 0
    alpha_vector = []
    i=0
    loss_list = []

    loss = torch.empty(0, requires_grad=True, device=device)
    for pred, mask, weight in zip(pred_list, mask_list, weight_list):
        if loss_fn=='iou':
            temp_loss = criterion(pred, mask, weight).view(1,1)
        elif loss_fn=='bce':
            # temp_loss = criterion(F.softmax(pred, dim=1), mask).mean(dim=[1,2,3]).view(-1,1)
            # temp_loss = (temp_loss*weight).sum().view(1,1)
            temp_loss = criterion(F.softmax(pred, dim=1), mask).mean(dim=[1], keepdim=True)
            # print('loss:', temp_loss.size(),'weight:', weight.size())
            temp_loss = (temp_loss*weight).sum().view(1,1)
        elif loss_fn=='ce':
            temp_loss = criterion(pred, mask).sum(dim=1)
            weight = weight.squeeze()
            # print('temp_loss:', temp_loss.size(), 'weight:', weight.size())
            # print('weighted_loss:', (temp_loss*weight).size())
            temp_loss = (temp_loss*weight).sum().view(1,1)

        loss_list.append(temp_loss.item())
        loss = torch.cat([loss, temp_loss], dim=1)

    epslion = 1/(level+1) #0.5
    if level==1:
        alpha_temp = (F.softmax(net.alpha_1.view(-1), dim=0)+epslion)
        alpha_temp = alpha_temp/alpha_temp.sum()
    elif level==2:
        alpha_temp = (F.softmax(net.alpha_2.view(-1), dim=0)+epslion)
        alpha_temp = alpha_temp/alpha_temp.sum()
    elif level==3:
        alpha_temp = (F.softmax(net.alpha_3.view(-1), dim=0)+epslion)
        alpha_temp = alpha_temp/alpha_temp.sum()
    elif level==4:
        alpha_temp = (F.softmax(net.alpha_4.view(-1), dim=0)+epslion)
        alpha_temp = alpha_temp/alpha_temp.sum()

    loss = torch.dot(loss.view(-1), alpha_temp)
    return loss, np.reshape(np.array(loss_list),[1,-1])
    
def train_net(net, device, train_dataset, val_dataset, epochs, batch_size, lr,level, 
            skip_option, loss_fn, dir_checkpoint, deep_sup, img_size, classes):
    n_train = train_dataset.__len__()
    n_val   = val_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    train_loss_list = []
    val_loss_list  = []
    train_iou_list = np.empty([0,5])
    val_iou_list  = np.empty([0,5])
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
        loss_fn:         {loss_fn}
    ''')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-8)
    train_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=1e-8)
    # train_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    if loss_fn=='iou':
        print('using iou loss.')
        criterion = mIoULoss(n_classes=classes).cuda()
    elif loss_fn =='bce':
        print('using Binary CrossEntropyLoss loss.')
        criterion = nn.BCELoss(reduction='none').cuda()
    elif loss_fn =='ce':
        print('using CrossEntropyLoss.')
        criterion = SoftCrossEntropy(reduction='none').cuda()

    count = level+1
    best_miou=0
    epslion = 1/(level+1) #0.5
    for epoch in range(epochs):
        if level==1:
            alpha_temp = (F.softmax(net.alpha_1.view(-1), dim=0).cpu().detach().numpy()+epslion)
            alpha_vector = alpha_temp/alpha_temp.sum()
        elif level==2:
            alpha_temp = (F.softmax(net.alpha_2.view(-1), dim=0).cpu().detach().numpy()+epslion)
            alpha_vector = alpha_temp/alpha_temp.sum()
        elif level==3:
            alpha_temp = (F.softmax(net.alpha_3.view(-1), dim=0).cpu().detach().numpy()+epslion)
            alpha_vector = alpha_temp/alpha_temp.sum()
        elif level==4:
            alpha_temp = (F.softmax(net.alpha_4.view(-1), dim=0).cpu().detach().numpy()+epslion)
            alpha_vector = alpha_temp/alpha_temp.sum()

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
            weight_list = [F.avg_pool2d(weight, kernel_size=2**i, stride=2**i)*((2**i)**2) for i in pool_size]+[weight]

            pred_list, _ = net(imgs)
            optimizer.zero_grad()
            loss, _ = compute_loss(criterion, pred_list, mask_list, net, weight_list, device, level, loss_fn)
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
        # if train_miou>best_miou:
        torch.save(net.state_dict(), f'{dir_checkpoint}{loss_fn}_{lr}_Ada_{skip_option}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')

    # draw_loss_curve(train_loss_list, val_loss_list, train_iou_list, val_iou_list, level, deep_sup, 'Ada_Ada', dir_checkpoint, lr, skip_option, loss_fn)

    train_loss_list = np.reshape(np.array(train_loss_list),[-1,1])
    val_loss_list   = np.reshape(np.array(val_loss_list),[-1,1])
    loss_curve = np.concatenate([train_loss_list, val_loss_list], axis=1)
    iou_curve = np.concatenate([train_iou_list, val_iou_list], axis=1)
    loss_record = pd.DataFrame(loss_curve)
    iou_record = pd.DataFrame(iou_curve)
    alpha_record = pd.DataFrame(alphas)
    train_h_loss = pd.DataFrame(train_h_loss)
    val_h_loss  = pd.DataFrame(val_h_loss)

    writer = pd.ExcelWriter(f'{dir_checkpoint}{loss_fn}_{lr}_Ada_{skip_option}_{level}.xlsx')

    loss_record.to_excel(writer, 'loss', float_format='%.8f')       
    iou_record.to_excel(writer, 'iou', float_format='%.8f')       
    alpha_record.to_excel(writer, 'alpha', float_format='%.8f')     
    train_h_loss.to_excel(writer, 'train_hidden', float_format='%.8f')
    val_h_loss.to_excel(writer, 'test_hidden', float_format='%.8f')
    writer.save()
    writer.close()
    return net, alpha_vector

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=60,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('-o', '--skip', dest='skip', type=str, default='none',
                        help='skip')
    parser.add_argument('-g', '--gpu', dest='n_gpu', type=int, default=0,
                        help='number of gpus')
    parser.add_argument('-c', '--loss', dest='loss', type=str, default='ce',
                        help='loss function')
    parser.add_argument('-d', '--data', dest='data', type=str, default='BCSS',
                        help='dataset')
    return parser.parse_args()

def get_data(train_sample_weight):
    train_img_pathes  = glob.glob(f'/ssd/BCSS-2021/train_imgs/*png')
    train_mask_pathes = glob.glob(f'/ssd/BCSS-2021/train_gts/*png')

    test_img_pathes   = glob.glob(f'/ssd/BCSS-2021/test_imgs/*png')
    test_mask_pathes  = glob.glob(f'/ssd/BCSS-2021/test_gts/*png')

    train_img_pathes.sort()
    train_mask_pathes.sort()
    test_img_pathes.sort()
    test_mask_pathes.sort()

    train_dataset = HistImageSet(train_img_pathes, train_mask_pathes, sample_weight=train_sample_weight, mask_channel=5)
    test_dataset   = HistImageSet(test_img_pathes, test_mask_pathes, sample_weight='mean', mask_channel=5)
    return train_dataset, test_dataset

def eval_UNet(net, val_dataset, device, batch_size, class_num, head):
    net.eval()
    error = 0
    sign_tensor = torch.empty(0,1,512,512)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            weight = batch['weight'].to(dtype=torch.float32)

            pred_list, etas = net(imgs)
            etas = normalize(etas)
            preds=0
            for i in range(0,len(etas)):
                preds += F.interpolate(F.softmax(pred_list[i],dim=1), size=(512,512), mode='bilinear', align_corners=True)*etas[i]
            probs = torch.argmax(preds, dim=1, keepdim=True)
            masks = torch.argmax(true_masks, dim=1, keepdim=True)
            sign = (probs!=masks).cpu()*1.0
            # print(sign_tensor.size(), sign.size())
            # print(sign_tensor, sign)
            sign_tensor = torch.cat((sign_tensor, 2*sign-1), dim=0)
            error += (sign*weight).sum()
    return error, sign_tensor

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

    dir_checkpoint = 'checkpoints/'+args.data+'/pixel_weight/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    train_sample_weight = 'mean'
    freeze=False
    for t in range(1,5):
        print(f'===========================iter:{t}==============================')
        net = AdaBoost_UNet(n_channels=n_channels, n_classes=classes, level=str(t), skip_option=args.skip, filters=24, deep_sup=deep_sup).cuda()

        if t==1:
            train_dataset, val_dataset = get_data('mean')
            # if os.path.exists(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada.pth'):
            #     print(f'loading weight from {dir_checkpoint}{args.loss}_{args.lr}_Ada.pth')
            #     net.load_state_dict(torch.load(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada.pth', map_location=device))

            for name,para in net.named_parameters():
                if name.split('.')[0] in ['X_00', 'X_10', 'up_10', 'out_01', 'out_10', 'alpha_1']:
                    para.requires_grad=True
                else:
                    para.requires_grad=False

        else:
            train_dataset, val_dataset = get_data(train_sample_weight)
            if os.path.exists(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}.pth'):
                print(f'loading weight from {dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}.pth')
                net.load_state_dict(torch.load(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}.pth', map_location=device))
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

        summary(net, (n_channels, img_size[0], img_size[1]))
        net.to(device=device)

        net, alpha_vector = train_net(net=net, device=device, train_dataset= train_dataset, val_dataset=val_dataset,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,level=t, 
                skip_option=args.skip, loss_fn=args.loss,dir_checkpoint=dir_checkpoint, 
                deep_sup=deep_sup, img_size=img_size, classes=classes)

        # net.load_state_dict(torch.load(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada.pth', map_location=device))
        head = np.argmax(alpha_vector)
        start_e = time.time()
        w_err, sign_tensor = eval_UNet(net, train_dataset, device, args.batch_size, classes, head)
        end_e = time.time()
        print(f'evaluation time: {round(end_e-start_e, 2)}, w_err:{round(w_err.item(),3)}, sign_tensor:{sign_tensor[:10,0,10,10]}')
        if w_err<(1-1/classes):
            freeze = True
            alpha=torch.log((1-w_err)/w_err)/2+torch.log(torch.tensor(classes-1, dtype=torch.float32))
            train_sample_weight = train_sample_weight*torch.exp(alpha*sign_tensor)
            train_sample_weight = train_sample_weight/torch.sum(train_sample_weight)

            sign_tensor = sign_tensor.cpu().numpy()
            w_max = np.max(sign_tensor)
            w_min = np.min(sign_tensor)
            weight_map = (sign_tensor-w_min)/(w_max-w_min)*255
            weight_map = weight_map.astype(np.uint8)
            
            if not os.path.exists(f'{dir_checkpoint}{args.skip}_map_{t}'):
                os.makedirs(f'{dir_checkpoint}{args.skip}_map_{t}')

            for i in range(weight_map.shape[0]):
                heatmap = cv2.applyColorMap(weight_map[i,0,:,:], cv2.COLORMAP_JET)
                cv2.imwrite(f'{dir_checkpoint}{args.skip}_map_{t}/{i}.png', heatmap)

        else:
            freeze = False
            alpha=torch.tensor(0)
        print('alpha_'+str(t)+':', alpha.item(), 'head:', head)
        alphas.append(str(head)+' '+str(alpha.item()))

    print('alphas:',alphas)
    end = time.time()
    print('running time:', end-start)
    f = open(f'{dir_checkpoint}{args.loss}_{args.lr}_Ada_{args.skip}.txt', 'w')
    for alpha in alphas:
        f.write(str(alpha))
        f.write("\n")
    # f.write('running time:'+str(end-start))
    f.close()