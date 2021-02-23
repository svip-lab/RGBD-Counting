from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


# return multi_apply(self.forward_single, feats)
def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func  # func
    # pfunc(forward_single)依次对*arg各个元素(即FPN的5个层级)进行forward
    # 返回包含每次pfunc函数返回值的新列表
    map_results = map(pfunc, *args)
    # print(map_results)  # map object
    results = tuple(map(list, zip(*map_results)))
    # print(results)
    return results


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
