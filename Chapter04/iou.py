import tensorflow as tf


def calculate_iou(gt_bb, pred_bb):
    '''
    :param gt_bb: ground truth bounding box
    :param pred_bb: predicted bounding box
    '''
    gt_bb = tf.stack([
        gt_bb[:, :, :, :, 0] - gt_bb[:, :, :, :, 2] / 2.0,
        gt_bb[:, :, :, :, 1] - gt_bb[:, :, :, :, 3] / 2.0,
        gt_bb[:, :, :, :, 0] + gt_bb[:, :, :, :, 2] / 2.0,
        gt_bb[:, :, :, :, 1] + gt_bb[:, :, :, :, 3] / 2.0])
    gt_bb = tf.transpose(gt_bb, [1, 2, 3, 4, 0])
    pred_bb = tf.stack([
        pred_bb[:, :, :, :, 0] - pred_bb[:, :, :, :, 2] / 2.0,
        pred_bb[:, :, :, :, 1] - pred_bb[:, :, :, :, 3] / 2.0,
        pred_bb[:, :, :, :, 0] + pred_bb[:, :, :, :, 2] / 2.0,
        pred_bb[:, :, :, :, 1] + pred_bb[:, :, :, :, 3] / 2.0])
    pred_bb = tf.transpose(pred_bb, [1, 2, 3, 4, 0])
    area = tf.maximum(
        0.0,
        tf.minimum(gt_bb[:, :, :, :, 2:], pred_bb[:, :, :, :, 2:]) -
        tf.maximum(gt_bb[:, :, :, :, :2], pred_bb[:, :, :, :, :2]))
    intersection_area= area[:, :, :, :, 0] * area[:, :, :, :, 1]
    gt_bb_area = (gt_bb[:, :, :, :, 2] - gt_bb[:, :, :, :, 0]) * \
                 (gt_bb[:, :, :, :, 3] - gt_bb[:, :, :, :, 1])
    pred_bb_area = (pred_bb[:, :, :, :, 2] - pred_bb[:, :, :, :, 0]) * \
                   (pred_bb[:, :, :, :, 3] - pred_bb[:, :, :, :, 1])
    union_area = tf.maximum(gt_bb_area + pred_bb_area - intersection_area, 1e-10)
    iou = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)
    return iou
