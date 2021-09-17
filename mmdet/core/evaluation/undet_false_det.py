from multiprocessing import Pool

import math
import numpy as np

from .bbox_overlaps import bbox_overlaps
# 漏检误检（同一类别下）
def upfp_singleclass(det_bboxes,
        gt_bboxes,
        img_shape,
        re_hw,
        stastic_undetect,
        file_name,
        gt_bboxes_ignore=None,
        iou_thr=0.5,
        high_score=0.15)->bool:
    """Check if the immage' gtboxes is undetected or false detected.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        high_score (float): bboxes whose score higher than this and no GT in this image,than undetected  
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        bool: whether undetected or falseDetected
    """
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    # print(re_hw)
    ch, cw = re_hw[0], re_hw[1]
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_gts = gt_bboxes.shape[0]
    num_dets = det_bboxes.shape[0]
    width, height = img_shape[1], img_shape[0]

    if num_gts > 0 :
        # 漏检1： 某张图片， 某个类别，有 GT 但是没有 dets
        if num_dets == 0: 
            if stastic_undetect:
                with open('/home/jingzhu/mmdet/cal.lst', 'a+') as f:
                    for gt_bbox in gt_bboxes:
                        w = gt_bbox[2] - gt_bbox[0]
                        h = gt_bbox[3] - gt_bbox[1]
                        ratio = 1.
                        if height > ch or width > cw:
                            ratio = np.min(np.array([ch, cw]).astype(np.float) / np.array([height, width]))
                        w *= ratio
                        h *= ratio
                        w, h = int(w), int(h)
                        square = math.sqrt(w*h)
                        f.write(str(w)+'*'+str(h)+'*'+str(w/h)[:3]+ '*' +str(square)[:3] + "*" + file_name +'\n')
            return True
        else:
        # 漏检2： 某张图片， 某个类别，GT 与 bbox 的 IOU 均小于某 threshold
            ious = bbox_overlaps(det_bboxes, gt_bboxes)
            mask = (ious.max(axis=0) < iou_thr)
            if mask.any():
                if stastic_undetect:
                    with open('/home/jingzhu/mmdet/cal.lst', 'a+') as f:
                        cout = np.where(mask==True)
                        cout = cout[0]
                        for out in cout:
                            gt_bbox = gt_bboxes[out]
                            w = gt_bbox[2] - gt_bbox[0]
                            h = gt_bbox[3] - gt_bbox[1]
                            ratio = 1.
                            if height > ch or width > cw:
                                ratio = np.min(np.array([ch, cw]).astype(np.float) / np.array([height, width]))
                            w *= ratio
                            h *= ratio
                            w, h = int(w), int(h)
                            square = math.sqrt(w*h)
                            f.write(str(w)+'*'+str(h)+'*'+str(w/h)[:3]+ '*' +str(square)[:3] + "*" + file_name +'\n')
                return True

    # 误检 1, 某张图片，某个类别，没有GT但是有置信度很高的bbox
    if num_gts == 0 and num_dets > 0 and (det_bboxes[:, -1] > high_score).any():
        return True
    return False


# 误检2: 某张图片，有GT，与GT iou 最高的bbox类别错误 而且具有较高 score
def iou_acrossclass(det_bboxes,
        gt_bboxes,
        gt_bboxes_ignore=None,
        high_score=0.15):
    """calculate iou across classes.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5). class j
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4). class i
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        max_iou (ndarray): max iou between class i 's GT boxes with class j 's det boxes. shape (n) 
    """
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))
    num_gts = gt_bboxes.shape[0]
    num_dets = det_bboxes.shape[0]

    if num_dets > 0:
        mask = det_bboxes[:, -1] > high_score
        det_bboxes = det_bboxes[mask,:]
        num_dets = det_bboxes.shape[0]
        if num_dets > 0:
            ious = bbox_overlaps(det_bboxes, gt_bboxes)
            return ious.max(axis=0)
    return np.zeros((num_gts))