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
    inter_width = max(0, inter_xa - inter_xb)
    inter_height = max(0, inter_ya - inter_yb)
    inter_area = inter_width * inter_height

    # compute the area of the two bounding boxes
    gt_box_ara = (xb_gt - xa_gt) * (yb_gt - ya_gt)
    pred_box_area = (xb_pred - xa_pred) * (yb_pred - ya_pred)

    # compute iou
    union_area = gt_box_ara + pred_box_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0

    return iou