import inspect
import mmcv


# model:
# from .registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS, ROI_EXTRACTORS, SHARED_HEADS)
# build_from_cfg(cfg.model, registry=DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
# ...
# dataset:
# from .registry import DATASETS
# dataset = build_from_cfg(cfg.data.train, registry=DATASETS, default_args)
def build_from_cfg(cfg, registry, default_args=None):
    """
    【Build a module from config dict.】

    Args:
        模型配置，模型占位符，训练/测试配置
        cfg (dict): Config dict. (It should at least contain the key "type".)
        registry (:obj:`Registry`): [The registry to search the type from.]
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The [constructed] object. - 构建完成的包含网络7个大类的模块

    注：上述7个大类，即便是DETECTORS，本质都是占位符，在传入cfg真正的参数之前都是不连接的，
        顺序是先搭建DETECTORS，然后根据其配置需求依次搭建其下的前几种模块，整个地构成DETECTORS
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None

    args = cfg.copy()  # cfg.model->args
    # print(arg)  # config info: RetinaNet->ResNeXt->FPN->FocalLoss->SmoothL1Loss...
    obj_type = args.pop('type')  # module name：RetinaNet/ResNeXt/FPN/FocalLoss/SmoothL1Loss/...
    # print('obj_type: ', obj_type)
    if mmcv.is_str(obj_type):
        # print(registry)
        # Registry(name=detector, items=['CascadeRCNN', 'TwoStageDetector', 'DoubleHeadRCNN', 'FastRCNN', 'FasterRCNN', 'SingleStageDetector', 'FCOS', 'GridRCNN', 'HybridTaskCascade', 'MaskRCNN', 'MaskScoringRCNN', 'RetinaNet', 'RPN'])
        # Registry(name=backbone, items=['ResNet', 'HRNet', 'ResNeXt', 'SSDVGG'])
        # Registry(name=neck, items=['BFP', 'FPN', 'HRFPN'])
        # ...
        # 这里的registry的get返回的_module_dict属性中包含的detector下的模型type
        # 索引key得到相应的class
        '''core'''
        obj_type = registry.get(obj_type)
        # print(obj_type)
        # <class 'mmdet.models.detectors.retinanet.RetinaNet'>
        # <class 'mmdet.models.backbones.resnext.ResNeXt'>
        # <class 'mmdet.models.necks.fpn.FPN'>
        # ...
        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)  # 将default_args的键值对加入到args中，将模型和训练配置整合送入类中

    # 注意，无论训练还是检测，都会build DETECTORS
    # **args将字典unpack得到各个元素分别与形参匹配送入函数中
    # print(obj_type(**args))
    # model
    '''首先从DETECTOR registry中引入RetinaNet[索引]，然后从这里执行前仍是class的索引，
    重点在于各个组件从**args处传入，然后便开始[初始化]各个模组类：
    首先是RetinaNet，super后进入SingleStageDetector，
    再从SingleStageDetector中build backbone、head等'''
    return obj_type(**args)

'''
功能：
    注册模块占位符
    在程序运行前先注册对应的模块占位，便于在config文件直接对应的模块进行配置填充
类型：
    7大类(实际后来dataset也是这么管理的)：BACKBONE,NECKS,ROI_EXTRACTORS,SHARED_HEADS,HEADS,LOSSES,DETECTORS
    每类包含各个具体的分类，如BACKBONES中有'ResNet','ResNeXt','SSDVGG'等
直观理解：
    Registry的具体形式是什么？
    例如import的DETECTOR直接打印可以得到Registry(name=detector, items=['CascadeRCNN', 'TwoStageDetector', 'DoubleHeadRCNN', 'FastRCNN', 'FasterRCNN', 'SingleStageDetector', 'FCOS', 'GridRCNN', 'HybridTaskCascade', 'MaskRCNN', 'MaskScoringRCNN', 'RetinaNet', 'RPN'])
    查看type为<class 'mmdet.utils.registry.Registry'>
    【Registry 7个类的模块每个下的_module_dict字典会添加存放其中的不同类
     - 作用是用于索引和搭建】
'''
# registry - DETECTORS = Registry('detector')
class Registry(object):
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()
        # print(self._module_dict) # {}

    # print info
    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    # obj_type = registry.get(obj_type='RetinaNet')
    def get(self, key):
        # print(self._module_dict)  # 包含了各个组件(detector,head...)的索引，每一类组成一个dict
        # class RetinaNet(SingleStageDetector)
        return self._module_dict.get(key, None)  # 获取class RetinaNet(见detectors:__init__.py)

    # 这里在mmdet/models/detectors/retinanet.py的class RetinaNet()
    # 装饰器@DETECTORS.register_module处调用传入config info
    # ->
    '''在__init___.py初始化时初始化class以调用装饰器修改_module_dict'''
    def register_module(self, cls):
        # print(cls)
        self._register_module(cls)
        return cls

    # self._register_module(cls)
    def _register_module(self, module_class):
        """
        【Register a module.】

        Args:
            module (:obj:[`nn.Module`]): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        # RetinaNet与single_stage都有
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        '''module_class继承自nn.module，为可训练类'''
        self._module_dict[module_name] = module_class


