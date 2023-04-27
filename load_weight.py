import os
import torch
import torch.nn as nn
from collections import OrderedDict
from unet.deep_supervise_unet_model import DS_UNet
from unet.new_Ada_DS import AdaBoost_UNet as AdaBoost_UNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
def load_weight(DS_weight, Ada_weight,fold):
	Ada_Net = AdaBoost_UNet(n_channels=1, n_classes=28, level='1234', filters=16, skip_option='scse')
	Ada_Net = nn.DataParallel(Ada_Net,device_ids=[0,1])
	Ada_Net.load_state_dict(torch.load(Ada_weight, map_location='cuda:0'))

	Ada_dict = Ada_Net.state_dict()
	Ada_para = Ada_Net.named_parameters()
	for k in Ada_dict:
		print(k)
	print('---------------')
	for name, para in Ada_para:
		print(name)

	DS_dict = torch.load(DS_weight)
	new_DS_dict = OrderedDict()
	source_keys = ['inc']+['down'+str(i) for i in range(1,5)]+\
				  ['up'+str(i) for i in range(1,5)]+\
				  ['out'+str(i) for i in range(1,9)]
	target_keys = ['X_'+str(i)+'0' for i in range(0,5)]+\
				  ['X_'+str(4-i)+str(i) for i in range(1,5)]+\
				  ['out_'+str(i)+'0' for i in range(1,5)]+['out_'+str(4-i)+str(i) for i in range(1,5)]
	# for s,t in zip(source_keys, target_keys):
	# 	print(s,'\t',t)
	for k, v in DS_dict.items():
		for s,t in zip(source_keys, target_keys):
			if s in k:
				new_DS_dict[k.replace(s,t)] = v
				break
	Ada_dict.update(new_DS_dict)
	Ada_Net.load_state_dict(Ada_dict)
	torch.save(Ada_Net.state_dict(),f'new_Ada_scse_{fold}.pth')

def check_weights(DS_weight, Ada_weight):
	DS_dict = torch.load(DS_weight, map_location='cpu')
	new_DS_dict = OrderedDict()
	source_keys = ['inc']+['down'+str(i) for i in range(1,5)]+\
				  ['up'+str(i) for i in range(1,5)]+\
				  ['out'+str(i) for i in range(1,9)]
	target_keys = ['X_'+str(i)+'0' for i in range(0,5)]+\
				  ['X_'+str(4-i)+str(i) for i in range(1,5)]+\
				  ['out_'+str(i)+'0' for i in range(1,5)]+['out_'+str(4-i)+str(i) for i in range(1,5)]
	for s,t in zip(source_keys, target_keys):
		print(s,'\t',t)
	for k, v in DS_dict.items():
		for s,t in zip(source_keys, target_keys):
			if s in k:
				new_DS_dict[k.replace(s,t)] = v
				break

	Ada_dict = torch.load(Ada_weight, map_location='cpu')
	for k in Ada_dict:
		if k.split('.')[1] in target_keys:
			if not (Ada_dict[k] == new_DS_dict[k]).all():
				print('Diff in {}'.format(k))
				print(f'Ada:{Ada_dict[k]}')
				print(f'DS:{new_DS_dict[k]}')



# folds=[str(i) for i in range(0,2)]
# for fold in folds:
# 	DS_weight=f'/home/yy3u19/mycode/Pytorch-UNet/checkpoints/Kylberg/avg_DS_UNet/DS_UNet_{fold}.pth'
# 	Ada_weight= f'checkpoints/Kylberg/new_Ada_scse_{fold}.pth'
	# load_weight(DS_weight, Ada_weight, fold)
	# check_weights(DS_weight, Ada_weight)

Ada_Net = AdaBoost_UNet(n_channels=1, n_classes=28, level='1234', filters=16, skip_option='scse')
for name, child in Ada_Net.named_children():
	for n, param in child.named_parameters():
		print(name+'.'+n)
