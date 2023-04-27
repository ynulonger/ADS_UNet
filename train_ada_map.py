import os
import sys
import time
import torch
import logging
import argparse
import numpy as np
import torchvision
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from metrics import *
from dice_loss import *
from torch import optim
from utils.dataset import *
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.ticker as ticker
from torch.autograd import Variable
from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from unet.Adaboost_UNet import AdaBoost_UNet as AdaBoost_UNet

def compute_loss(criterion, pred, mask, loss_fn, sample_weight):
    loss = 0
    if loss_fn=='bce':
        loss = criterion(F.softmax(pred, dim=1), mask)
        if sample_weight!=None:
            loss = loss*sample_weight
            loss = loss.sum()
    elif loss_fn=='ce':
        loss = criterion(pred, mask).sum()
    elif loss_fn=='iou':
        loss = criterion(pred, mask, sample_weight)
    return loss
    # print(loss.size(), sample_weight.size())


def evaluate(criterion, net, loader, img_size, device, classes, loss_fn):
    net.eval()
    metrics = Metrics(classes, ignore_index=11)
    gt = torch.empty([0, img_size[0],img_size[1]]).to(device=device, dtype=torch.float32)
    predictions  = torch.empty([0, img_size[0], img_size[1]]).to(device=device, dtype=torch.float32)
    iou = 0
    loss = 0
    length = 0

    for batch in loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            weight = batch['weight'].to(device=device, dtype=torch.float32)

            masks_pred = net(imgs)
            temp_loss = compute_loss(criterion, masks_pred, true_masks, loss_fn, weight)

            loss += temp_loss.item()
            preds = F.softmax(masks_pred,dim=1)
            preds = torch.argmax(preds, dim=1).squeeze(1)
            if len(true_masks.size())==4:
                true_masks=torch.argmax(true_masks, dim=1)
            masks = true_masks.view(-1)
            preds = preds.view(-1)
            metrics.add(preds, masks)
            length += 1
    miou = metrics.iou(average=True)
    metrics.clear()
    return miou, loss
    
def train_net(net, device, train_dataset, val_dataset, epochs, batch_size, lr,level, skip_option, loss_fn, dir_checkpoint, deep_sup, img_size, classes):
    n_train = train_dataset.__len__()
    n_val   = val_dataset.__len__()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    train_loss_list = []
    val_loss_list  = []
    train_iou_list = []
    val_iou_list = []
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        skip option      {skip_option}
    ''')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5, gamma=0.5)
    train_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)


    if loss_fn=='iou':
        print('using iou loss.')
        criterion = mIoULoss(n_classes=classes, ignore_indix=11).cuda()
    elif loss_fn =='bce':
        criterion = nn.BCELoss(reduction='none').cuda()
    elif loss_fn =='ce':
        criterion = nn.CrossEntropyLoss(reduction='sum').cuda()

    count = int(level)+1
    best_miou = 0
    for epoch in range(epochs):
        e_start=time.time()
        learning_rate = train_scheduler.get_last_lr()
        net.train()
        train_loss = 0
        for batch in train_loader:
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            true_masks = Variable(batch['mask'].to(device=device, dtype=torch.float32))
            weight = batch['weight'].to(device=device, dtype=torch.float32)

            pred_list = net(imgs)
            optimizer.zero_grad()
            loss = compute_loss(criterion, pred_list, true_masks, loss_fn, weight)
            # train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_scheduler.step()

        train_miou, train_loss = evaluate(criterion, net, train_loader, img_size, device, classes, loss_fn)
        val_miou, val_loss = evaluate(criterion, net, val_loader, img_size, device, classes, loss_fn)

        e_end=time.time()
        # val_loss = val_loss/(img_size[0]*img_size[1])
        print('Epoch:', epoch+1,'/',epochs, 'time:',e_end-e_start,'Lr:', learning_rate, 'Train_Loss:', train_loss, 'Val_Loss:', val_loss, 'Train_miou:', train_miou, 'val_miou:',val_miou)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_iou_list.append(train_miou)
        val_iou_list.append(val_miou)

        if train_miou>best_miou:
            best_miou = train_miou
            torch.save(net.state_dict(),
                    f'{dir_checkpoint}Ada_{loss_fn}_{skip_option}_{deep_sup}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    draw_loss_curve(train_loss_list, val_loss_list, train_iou_list, val_iou_list, level, deep_sup, 'ada', dir_checkpoint, loss_fn)

    train_loss_list = np.reshape(np.array(train_loss_list),[-1,1])
    val_loss_list   = np.reshape(np.array(val_loss_list),[-1,1])
    loss_curve = np.concatenate([train_loss_list, val_loss_list], axis=1)
    loss_record = pd.DataFrame(loss_curve)
    writer = pd.ExcelWriter(f'{dir_checkpoint}excels/{deep_sup}_Ada_{loss_fn}_{skip_option}_{level}.xlsx')

    loss_record.to_excel(writer, 'loss', float_format='%.8f')       
    writer.save()
    writer.close()
    return net

def draw_loss_curve(train_loss, val_loss, train_iou, val_iou, level, deep_sup, model, dir_checkpoint, loss_fn):

    x    = [i+1 for i in range(len(train_loss))]
    fig  = plt.figure(figsize=(12,6))
    loss = fig.add_subplot(111)

    loss_train=loss.plot(x, train_loss, linestyle='--', label='Train loss')
    loss_test =loss.plot(x, val_loss, linestyle='--', label='Test loss')

    mIoU = loss.twinx()
    mIoU_train=mIoU.plot(x, train_iou, label='Train mIoU')
    mIoU_test =mIoU.plot(x, val_iou, label='Test mIoU')

    lns = loss_train+loss_test+mIoU_train+mIoU_test
    labs = [l.get_label() for l in lns]

    loss.legend(lns, labs, loc = "upper left")
    loss.set_xlabel("Epoch")
    loss.set_ylabel("Loss")
    mIoU.set_ylabel("mIoU")
    plt.savefig(f'{dir_checkpoint}figs/{deep_sup}_Ada_{loss_fn}_{level}.png')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=70,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, 
                        help='batch size')
    parser.add_argument('-o', '--skip', dest='skip', type=str, default='scse',
                        help='skip')
    parser.add_argument('-g', '--gpu', dest='n_gpu', type=int, default=0,
                        help='number of gpus')
    parser.add_argument('-c', '--loss', dest='loss', type=str, default='bce',
                        help='loss function')
    parser.add_argument('-d', '--data', dest='data', type=str, default=0,
                        help='dataset')
    return parser.parse_args()

def get_data(dataset, sample_weight):
    if dataset=='prague':
        train_img_pathes = glob.glob('/ssd/Custom_data/train/imgs/*')
        train_mask_pathes = glob.glob('/ssd/Custom_data/train/gts/*')
        test_img_pathes = glob.glob('/ssd/Custom_data/test/imgs/*')
        test_mask_pathes = glob.glob('/ssd/Custom_data/test/gts/*')

        train_img_pathes.sort()
        train_mask_pathes.sort()
        test_img_pathes.sort()
        test_mask_pathes.sort()

        train_dataset = Prague(train_img_pathes, train_mask_pathes, sample_weight)
        val_dataset   = Prague(test_img_pathes, test_mask_pathes, sample_weight='mean')

    elif dataset == 'camvid':
        train_dataset = CamVid('train+val', sample_weight=sample_weight)
        val_dataset   = CamVid('test', sample_weight='mean')

    return train_dataset, val_dataset

def eval_UNet(net, val_dataset, device, batch_size, img_size, class_num, iteration):
    net.eval()
    save_patches = 'data/camvid/Ada/'
    if not os.path.exists(save_patches):
        os.makedirs(save_patches)

    metrics = Metrics(class_num, ignore_index=11)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    weighted_acc = torch.empty(0)
    mIoU_list = []
    weighted_iou = 0
    error_list = []
    weighted_error = 0

    new_sample_weight = []
    for batch in val_loader:
        with torch.no_grad():
            imgs = Variable(batch['image'].to(device=device, dtype=torch.float32))
            masks = batch['mask'].to(device=device, dtype=torch.float32)
            names = batch['name']
            weight = batch['weight'].to(device=device, dtype=torch.float32)
            output = F.softmax(net(imgs), dim=1)

            preds = output.argmax(dim=1)
            if len(masks.size())==4:
                masks=torch.argmax(masks, dim=1)

            for i in range(preds.shape[0]):
                pred = preds[i]
                mask = masks[i]
                error = (pred!=mask)+1.0
                error = error.view(1,img_size[0], img_size[1])
                error_blur = TF.gaussian_blur(error, [7,7], sigma=2.0)
                heatmap = cv2.applyColorMap((error_blur.squeeze()*127.5).cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
                # print(names[i])
                cv2.imwrite(save_patches+str(iteration)+'_'+names[i], heatmap)
                error_blur = error_blur/error_blur.sum()
                # print(new_sample_weight.size(), error_blur.size())
                new_sample_weight.append(error_blur.cpu())

            masks = masks.view(-1)
            preds = preds.view(-1)

            metrics.add(preds, masks)
    miou = metrics.iou(average=False)
    metrics.clear()
    return miou, new_sample_weight

if __name__ == '__main__':
    img_size=[360, 480]
    deep_sup=False
    n_channels = 3
    start = time.time()
    classes = 12
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

    # if args.data=='hist':
    #     dir_checkpoint = 'checkpoints/hist/'
    # elif args.data=='forest':
    #     dir_checkpoint = 'checkpoints/forest/'

    print('-------------------------', args.loss)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    alphas = []

    dir_checkpoint = 'checkpoints/'+args.data+'/'

    sample_weight='mean'
    for t in range(1,5):
        print(f'===========================iter:{t}==============================')
        net = AdaBoost_UNet(n_channels=3, n_classes=classes, level=str(t), skip_option=args.skip, filters=64, deep_sup=deep_sup).cuda()
        train_dataset, val_dataset = get_data(args.data, sample_weight) 

        if t==1:
            for name,para in net.named_parameters():
                if 'X_00' in name or 'X_10' in name or 'up_10' in name or 'out_01' in name or 'out_10' in name:
                    para.requires_grad=True
                else:
                    para.requires_grad=False

        if t !=1:
            print(f'loading weight from {dir_checkpoint}Ada_{args.loss}_{args.skip}.pth')
            net.load_state_dict(torch.load(f'{dir_checkpoint}Ada_{args.loss}_{args.skip}_{deep_sup}.pth', map_location=device))
            depth=t+1
            
            if freeze:
                up_list = []
                out_list= []
                for i in range(depth):
                    temp = 'out_'+str(i)+str(t-i)
                    out_list.append(temp)
                idx_1 = [i for i in range(t)]
                idx_2 = [i for i in range(1, depth)]
                idx_2.reverse()
                for i,j in zip(idx_2, idx_1):
                    temp = 'up_'+str(i)+str(j)
                    up_list.append(temp)

                for name,para in net.named_parameters():
                    if name[:5] in up_list:
                        para.requires_grad=True
                    elif name[:6] in out_list:
                        para.requires_grad=True
                    elif name[:4] == 'X_'+str(t)+'0':
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
                    else:
                        para.requires_grad=False

        # for name,para in net.named_parameters():
        #     if para.requires_grad==True:
        #         print(f'Trainable weights: {name}')

        summary(net, (n_channels, img_size[0], img_size[1]))
        net.to(device=device)

        net = train_net(net=net, device=device, train_dataset= train_dataset, val_dataset=val_dataset,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,level=str(t), skip_option=args.skip, 
                        loss_fn=args.loss,dir_checkpoint=dir_checkpoint, deep_sup=deep_sup, img_size=img_size, classes=classes)

        start_e = time.time()
        IoU_list, sample_weight = eval_UNet(net, train_dataset, device, args.batch_size, img_size, classes, iteration=str(t))
        end_e = time.time()
        print('evaluation time:', end_e-start_e)
        print(f'IoU:{IoU_list.mean()}')
        w_error = 1-IoU_list.mean()
        if w_error<0.5:
            freeze = True
            alpha=torch.log((1-w_error)/w_error)/2
            print('alpha_'+str(t)+':', alpha)
            alphas.append(alpha)
        else:
            freeze = False
            alpha=torch.tensor(0)
            print('alpha_'+str(t)+':', alpha)
            alphas.append(alpha)

    print('alphas:',alphas)
    end = time.time()
    print('running time:', end-start)
    f = open(f'{dir_checkpoint}{deep_sup}_{args.loss}_{args.skip}.txt', 'w')
    for alpha in alphas:
        f.write(str(alpha.item()))
        f.write("\n")
    # f.write('running time:'+str(end-start))
    f.close()


