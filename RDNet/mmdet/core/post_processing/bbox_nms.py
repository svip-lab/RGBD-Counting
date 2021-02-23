import torch

from mmdet.ops.nms import nms_wrapper


#【bboxes从大坐标变成了较小的数？过程】
# det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
#                    cfg.score_thr=0.05, cfg.nms=dict(type='nms',iou_thr=0.5), cfg.max_per_img=100)
def multiclass_nms(multi_bboxes, multi_scores,
                   score_thr, nms_cfg, max_num=-1, score_factors=None):
    """[NMS for multi-class bboxes].

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class) - 1+1/80+1
        score_thr (float): bbox threshold(bboxes with scores lower than it
            will not be considered).
        nms_thr (float): NMS IoU threshold
        max_num (int): (if there are more than max_num bboxes after NMS),
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels are 0-based.
    """
    num_classes = multi_scores.shape[1]  # 2
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)  # <function nms at 0x7f5de7ae5158>
    # 对于每一类：
    for i in range(1, num_classes):  # 第0维是padding
        # ！！！！！！！！！！！！！！！！
        cls_inds = multi_scores[:, i] > score_thr  # 取proposal中第i类score大于score_thr的索引
        if not cls_inds.any():  # all score lower than score_thr for i-th class
            # print('no object found')
            continue
        # 后面在RGBD中不会进入if score_thr=0.05
        # print('object found')  # found i-th class object
        # ->
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]  # 取上述索引指向的proposal
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]  # 取score
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        # ->
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)  # 【cls_dets <- bboxes+scores】
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0],),  # 【cls_labels <- i-1】
                                           i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)

    if bboxes:
        # √
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:  # NMS
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]  # 取分数最高的max_num个
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
