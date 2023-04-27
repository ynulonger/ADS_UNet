import torch
from unet.deep_supervise_unet_model import DS_UNet
from unet.new_Ada_DS import AdaBoost_UNet as AdaBoost_UNet

def load_weight(DS_weight, Ada_weight):
	DS_UNet = DS_UNet(n_channels=1, n_classes=28, filters=16, bilinear=False).cuda()
	Ada_Net = AdaBoost_UNet(n_channels=1, n_classes=28, level='1234', filters=16, skip_option='scse').cuda()

	Ada_dict = Ada_Net.state_dict()
	DS_dict = torch.load(DS_weight)
	new_DS_dict = OrderedDict()
	
	source_keys = ['inc']+['down'+str(i) for i in range(1,5)]+['up'+str(i) for i in range(1,5)]+['out'+str(i) for i in range(1,9)]
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
	Ada_dict.update(new_DS_dict)
	Ada_Net.load_state_dict(Ada_dict)
	torch.save(Ada_Net.state_dict(),Ada_weight)

DS_weight='/home/yy3u19/mycode/Pytorch-UNet/checkpoints/Kylberg/avg_DS_UNet/DS_UNet_0.pth'
Ada_weight='new_Ada_scse_0.pth'


