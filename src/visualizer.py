import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.metrics import compute_detection_metrics


def visualize_few_samples(dataset_df: pd.DataFrame, n_samples: int = 5, random_state: int = 42) -> None:
    """
    Function that visualizes few samples from the dataset, plotting the image and the prediction and ground truth
    bounding boxes.

    Parameters
    ----------
    dataset_df: pd.DataFrame
        The dataframe representing the dataset to visualize.
    n_samples: int
        The number of samples to draw from the dataset (defaults to 5).
    random_state: int
        The seed of the random number generator (defaults to 42).

    Returns
    -------
    None
    """

    # sample some elements from the dataframe to visualize
    samples_to_visualize = dataset_df.sample(n=n_samples, random_state=random_state)

    # plot the samples image with the bounding boxes
    for index, sample in samples_to_visualize.iterrows():
        plot_boxes(sample["image-data"],
                   (sample['gt-x-start'], sample['gt-x-end'], sample['gt-y-start'], sample['gt-y-end']),
                   (sample['pred-x-start'], sample['pred-x-end'], sample['pred-y-start'],
                    sample['pred-y-end']))


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
    plt.figure(figsize=(8, 8))
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
    """
    Function that plots the IoU distribution of a dataset.

    Parameters
    ----------
    dataset_df: pd.DataFrame
        The dataframe representing the dataset.
    class_type: str
        The class type of the samples to consider in the plot (default is None).

    Returns
    -------
    None
    """

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
    """
    Function that prints a detection report for a given dataset.

    Parameters
    ----------
    dataset_df: pd.DataFrame
        The dataframe to use for the detection report.
    class_type: str
        The class type to consider in the plot data (defaults to None).

    Returns
    -------
    None
    """

    # compute detection metric for the given dataset
    accuracy, precision, recall, f1, tp, fp, tn, fn, errors = compute_detection_metrics(dataset_df=dataset_df)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Total samples: {len(dataset_df)}')
    print(f'True Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'True Negatives: {tn}')
    print(f'False Negatives: {fn}')
    for key, value in errors.items():
        print(f'{key}: {value:.2f}')

    plot_iou_distribution(dataset_df=dataset_df, class_type=class_type)
