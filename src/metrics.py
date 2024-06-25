import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from typing import Tuple


def iou(gt_box: tuple, pred_box: tuple) -> float:
    """
    Function computing the intersection over union between two bounding boxes.

    Parameters
    ----------
    gt_box: tuple
        A tuple containing the coordinates of the top left and the bottom right corner of the ground truth bounding box.
    pred_box: tuple
        A tuple containing the coordinates of the top left and the bottom right corner of the predicted bounding box.

    Returns
    -------
    float
        The intersection over union score of the two bounding boxes.
    """

    # unpack coordinates of the two boxes
    xa_gt, xb_gt, ya_gt, yb_gt = gt_box
    xa_pred, xb_pred, ya_pred, yb_pred = pred_box

    # compute the coordinates of the intersection rectangle
    inter_xa = max(xa_gt, xa_pred)
    inter_ya = max(ya_gt, ya_pred)
    inter_xb = min(xb_gt, xb_pred)
    inter_yb = min(yb_gt, yb_pred)

    # compute the area of the intersection rectangle
    inter_width = max(0, inter_xb - inter_xa)
    inter_height = max(0, inter_yb - inter_ya)
    inter_area = inter_width * inter_height

    # compute the area of the two bounding boxes
    gt_box_ara = (xb_gt - xa_gt) * (yb_gt - ya_gt)
    pred_box_area = (xb_pred - xa_pred) * (yb_pred - ya_pred)

    # compute iou
    union_area = gt_box_ara + pred_box_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def compute_detection_predictions(dataset_df: pd.DataFrame, detection_iou_threshold: float = 0.5) -> pd.DataFrame:
    # compute IoU for all samples in the dataset, adding a column to the dataframe
    dataset_df['iou'] = dataset_df.apply(lambda row: iou(
        (row['gt-x-start'], row['gt-x-end'], row['gt-y-start'], row['gt-y-end']),
        (row['pred-x-start'], row['pred-x-end'], row['pred-y-start'], row['pred-y-end'])
    ), axis=1)

    # the sample is detected if the IoU between the predicted and the ground truth bounding box is greater than the
    # threshold
    dataset_df['pred-detected'] = dataset_df['iou'] > detection_iou_threshold

    return dataset_df


def compute_detection_metrics(dataset_df: pd.DataFrame) -> Tuple:
    # select detection predictions
    y_pred = dataset_df['pred-detected']

    # define the array of detection ground truths; since each sample in the dataset represents a nodule that should have
    # been detected, the array is going to be an array of true values with the same number of samples as in the dataset
    y_true = np.full(len(dataset_df), True)

    # compute accuracy, precision, recall and f1 scores
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1
