# 先看__init__.py就能大致得知引入了哪些类型loss可供选择，以及通过哪些文件配置以便底层更改
from .accuracy import Accuracy, accuracy
from .balanced_l1_loss import BalancedL1Loss, balanced_l1_loss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .focal_loss import FocalLoss, sigmoid_focal_loss
from .ghm_loss import GHMC, GHMR
from .iou_loss import BoundedIoULoss, IoULoss, bounded_iou_loss, iou_loss
from .mse_loss import MSELoss, mse_loss
from .smooth_l1_loss import SmoothL1Loss, smooth_l1_loss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'sigmoid_focal_loss',
    'FocalLoss', 'smooth_l1_loss', 'SmoothL1Loss', 'balanced_l1_loss',
    'BalancedL1Loss', 'mse_loss', 'MSELoss', 'iou_loss', 'bounded_iou_loss',
    'IoULoss', 'BoundedIoULoss', 'GHMC', 'GHMR', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss'
]

# 构建时统一采用registry进行管理，
# 所以子函数方法会调用静态修饰器提前把loss实现类的函数名传递到字典中去，
# 该loss的类实际以继承自nn.Module的层的形式进行封装，
# 通过执行前向传播实现loss计算，在forward中定义

# 调用方法：
# 首先是head初始化的build进行组件堆叠时，调用loss层的构造函数__init__，
# 初始化loss层作为一个层被加进去叠上，返回一个layer的类
# 在进行前向传播时，通过传入计算参数调用forward