import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from metrics import compute_detection_metrics


def plot_boxes(image: np.ndarray, gt_box: tuple, pred_box: tuple) -> None:
    """
    Method that plots the ground truth and predicted bounding boxes on the image.

    Parameters
    ----------
    image: np.ndarray
        A numpy array representing the image.
    gt_box: tuple
        A tuple containing the coordinates of the top left and the bottom right corner of the ground truth bounding box.
    pred_box: tuple
        A tuple containing the coordinates of the top left and the bottom right corner of the predicted bounding box.

    Returns
    -------
    Nothing
    """

    # unpack the bounding boxes coordinates
    xa_gt, xb_gt, ya_gt, yb_gt = gt_box
    xa_pred, xb_pred, ya_pred, yb_pred = pred_box

    # plot the image
    plt.figure(figsize=(16, 16))
    plt.imshow(image, cmap='gray')

    # plot ground truth bounding box in red
    gt_rect = plt.Rectangle((xa_gt, ya_gt), xb_gt - xa_gt, yb_gt - ya_gt, edgecolor='red', facecolor='none',
                            linewidth=1)
    plt.gca().add_patch(gt_rect)

    # plot predicted bounding box in green
    pred_rect = plt.Rectangle((xa_pred, ya_pred), xb_pred - xa_pred, yb_pred - ya_pred, edgecolor='green',
                              facecolor='none', linewidth=1)
    plt.gca().add_patch(pred_rect)

    plt.axis('off')
    plt.show()


def plot_iou_distribution(dataset_df: pd.DataFrame, class_type: str = None) -> None:
    # plot the IoU distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset_df['iou'], bins=30, kde=True)
    if class_type:
        plt.title(f'IoU distribution for class "{class_type}"')
    else:
        plt.title(f'IoU distribution')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.show()


def detection_report(dataset_df: pd.DataFrame, class_type: str = None) -> None:
    accuracy, precision, recall, f1 = compute_detection_metrics(dataset_df=dataset_df)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    plot_iou_distribution(dataset_df=dataset_df, class_type=class_type)
