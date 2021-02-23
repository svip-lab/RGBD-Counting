from __future__ import print_function

import numpy as np
import json
import os
import torch

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih
    return intersection / ua

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(results, annotations, checkpoint, iou_threshold=0.5, save_path=None):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        results         : detection results
        annotations     : original data
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    all_detections = results
    all_annotations = annotations

    average_precisions = {}
    for label in range(len(all_detections[0])):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_detections)):
            # 这里对应到outputs的全部检测框
            detections = all_detections[i][label]
            # 这里应对应annotations的全部gt
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations.numpy())
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        # recall
        recall = true_positives / num_annotations
        # precision
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('\nmAP:')
    print(average_precisions[0][0])
    MAE, MSE = calculate_mae_mse(all_annotations, all_detections)
    print('\nMAE: {0}\tMSE: {1}'.format(MAE, MSE))
    checkpoint = checkpoint[checkpoint.rfind('_') + 1:]
    print(checkpoint)

    # return average_precisions, MAE, MSE
    return average_precisions


def calculate_mae_mse(all_annotations, all_detections):
    '''
    calculate MAE and MSE for crowd counting
    inputs: all_annotations, all_detections (list of list)
    outputs: MAE and MSE
    ''' 
    assert len(all_annotations) == len(all_detections) 
    gt_crowd_num = []
    pre_crowd_num = []
    for img_id in range(len(all_annotations)):
        assert len(all_annotations[img_id]) == 1
        assert len(all_detections[img_id]) == 1
        gt_crowd_num.append(all_annotations[img_id][0].shape[0])
        pre_crowd_num.append(all_detections[img_id][0].shape[0])

    gt_crowd_num = np.array(gt_crowd_num)
    pre_crowd_num = np.array(pre_crowd_num)

    error = gt_crowd_num - pre_crowd_num
    MAE = abs(error).sum()/error.shape[0]
    MSE = np.sqrt((error ** 2).sum()/error.shape[0])
    return MAE, MSE







