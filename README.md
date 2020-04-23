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
Download our ShanghaiTechRGBD dataset: [OneDrive](https://yien01-my.sharepoint.com/:f:/g/personal/doubility_z0_tn/EhY4Svr1rRlDi7apZTtpepQBJejNSSYnQk1UNSqxhQ3jqA?e=RdhCtz)

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
Some explanations of data preprocessing:

The metric of depth is millimeter. The max covered depth value of depth sensor is 20 meters, which means the normal range of depth value should be 0-20,000. -999 means the depth value at this point is out of range. Only a few points are above 20,000 in an depth image, which can be the inherent problem of depth sensor resulted from occlusion or illumination, etc. Therefore, we cut off [0, 20,000] for the whole depth image as the valid depth value and set values at other pixels as 30,000. It is an empirical setting through the observing for image. After that, we normalize the depth image through dividing 20,000. 


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