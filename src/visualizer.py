import numpy as np
import matplotlib.pyplot as plt


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
