# RGBD crowd counting 

PyTorch implementation of our CVPR2019 paper:

**'Density Map Regression Guided Detection Network for RGB-D Crowd Counting and Localization'** [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lian_Density_Map_Regression_Guided_Detection_Network_for_RGB-D_Crowd_Counting_CVPR_2019_paper.pdf)]
[[poster](images/poster.pdf)]

Dongze Lian*, Jing Li*, Jia Zheng, Weixin Luo, Shenghua Gao
 
(* Equal Contribution)

# Requirements
- python: 3.x
- Pytorch: 0.4+
- torchvision: 0.2+

# The ShanghaiTechRGBD dataset
Download our ShanghaiTechRGBD dataset: [OneDrive](https://yien01-my.sharepoint.com/:f:/g/personal/doubility_z0_tn/EhY4Svr1rRlDi7apZTtpepQBHI7fsLFsclR0t4G-q6ugtA?e=zHVs5z) or [BaiduPan](https://pan.baidu.com/s/1Loet9ekp_oYD1xT7y7wv5g) (Code: 5luu)

```
ShanghaiTechRGBD/
├── train_data/
    ├── train_img/*.png
    ├── train_depth/*.mat
    └── train_gt/*.mat
└── test_data/
    ├── test_img/*.png
    ├── test_depth/*.mat
    └── test_bbox_anno/*.mat
```

# Usage

Code is coming soon.


# Citation

```
@InProceedings{Lian_2019_CVPR,
author = {Lian, Dongze and Li, Jing and Zheng, Jia and Luo, Weixin and Gao, Shenghua},
title = {Density Map Regression Guided Detection Network for RGB-D Crowd Counting and Localization},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```