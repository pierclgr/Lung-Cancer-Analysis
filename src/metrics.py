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


def compute_iou_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that adds an IoU column to a DataFrame representing an object detection dataset.

    Parameters
    ----------
    dataset_df: pd.DataFrame
        The dataset that contains the object detection predicted and ground truth boxes.

    Returns
    -------
    pd.DataFrame
        The dataset with the IoU column added to it.
    """

    # compute IoU for all samples in the dataset, adding a column to the dataframe
    dataset_df['iou'] = dataset_df.apply(lambda row: iou(
        (row['gt-x-start'], row['gt-x-end'], row['gt-y-start'], row['gt-y-end']),
        (row['pred-x-start'], row['pred-x-end'], row['pred-y-start'], row['pred-y-end'])
    ), axis=1)

    return dataset_df


def compute_detection_predictions(dataset_df: pd.DataFrame, detection_iou_threshold: float = 0.5) -> pd.DataFrame:
    """
    Function that computes detection predictions accordingly to a given IoU threshold.

    Parameters
    ----------
    dataset_df: pd.Dataframe
        The DataFrame of the dataset that contains the object detection predictions and ground truths.
    detection_iou_threshold: float
        The IoU threshold to use for computing the detection predictions (defaults to 0.5).

    Returns
    -------
    pd.DataFrame
        The dataframe with the predictions of the detections added.
    """

    # the sample is detected if the IoU between the predicted and the ground truth bounding box is greater than the
    # threshold
    df_copy = dataset_df.copy()
    df_copy['pred-detected'] = dataset_df['iou'] > detection_iou_threshold

    return df_copy


def compute_detection_metrics(dataset_df: pd.DataFrame) -> Tuple:
    """
    Function that computes detection metrics using the given DataFrame representing the dataset.

    Parameters
    ----------
    dataset_df: pd.DataFrame
        The DataFrame of the dataset that contains the object detection predictions and ground truths.

    Returns
    -------
    tuple
        A tuple containing
        - accuracy
        - precision
        - recall
        - f1
        - number of true positives
        - number of false positives
        - number of true negatives
        - number of false negatives
        - a dictionary containing bounding boxes errors averages
    """

    # select detection predictions
    y_pred = dataset_df['pred-detected']

    # define the array of detection ground truths; since each sample in the dataset represents a nodule that should have
    # been detected, the array is going to be an array of true values with the same number of samples as in the dataset
    y_true = np.full(len(dataset_df), True)

    # compute the number of TP, FP and FN
    # True Positives (TP)
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # False Positives (FP)
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # True Negatives (TN)
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # False Negatives (FN)
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # compute accuracy, precision, recall and f1 scores
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    samples = dataset_df[dataset_df["pred-detected"]]

    # calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE) for bounding box coordinates
    mae_x_start = np.mean(np.abs(samples['gt-x-start'] - samples['pred-x-start']))
    mae_x_end = np.mean(np.abs(samples['gt-x-end'] - samples['pred-x-end']))
    mae_y_start = np.mean(np.abs(samples['gt-y-start'] - samples['pred-y-start']))
    mae_y_end = np.mean(np.abs(samples['gt-y-end'] - samples['pred-y-end']))

    mse_x_start = np.mean((samples['gt-x-start'] - samples['pred-x-start']) ** 2)
    mse_x_end = np.mean((samples['gt-x-end'] - samples['pred-x-end']) ** 2)
    mse_y_start = np.mean((samples['gt-y-start'] - samples['pred-y-start']) ** 2)
    mse_y_end = np.mean((samples['gt-y-end'] - samples['pred-y-end']) ** 2)

    errors = {'Mean Absolute Error (MAE) for x_start': mae_x_start,
              'Mean Absolute Error (MAE) for x_end': mae_x_end,
              'Mean Absolute Error (MAE) for y_start': mae_y_start,
              'Mean Absolute Error (MAE) for y_end': mae_y_end,

              'Mean Squared Error (MSE) for x_start': mse_x_start,
              'Mean Squared Error (MSE) for x_end': mse_x_end,
              'Mean Squared Error (MSE) for y_start': mse_y_start,
              'Mean Squared Error (MSE) for y_end': mse_y_end}

    return accuracy, precision, recall, f1, tp, fp, tn, fn, errors
