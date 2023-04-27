# ADS_UNet: A nested UNet for Histopathology Image Segmentation

This repository is the official PyTorch implementation of the Expert System With Application (ESWA) paper "ADS_UNet: A nested UNet for Histopathology Image Segmentation" by Yilong Yang, Srinandan Dasmahapatra and Sasan Mahmoodi.

```
@article{yang2023ads_unet,
  title={ADS\_UNet: A nested UNet for histopathology image segmentation},
  author={Yang, Yilong and Dasmahapatra, Srinandan and Mahmoodi, Sasan},
  journal={Expert Systems with Applications},
  year={2023},
  publisher={Elsevier}
}
```

### Data Organization

```
ADS_UNet
|--utils
    |--utils such as loss function, dataloader, etc...
|--unet
    |--implementation of UNet, UNet++, ADS_UNet, etc...
|--checkpoints
    |--checkpoints of trained models
|--.py files
```

### Instructions
#### Dataset preparation.
   Download BCSS, CRAG, and MoNoSeg datasets and change the paths of these datasets in utils/dataset.py accordingly.