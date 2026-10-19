"""[hailo-test] Pure-numpy drop-in for cython_bbox.bbox_overlaps (same +1 pixel convention)."""
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    query_boxes = np.asarray(query_boxes, dtype=np.float64).reshape(-1, 4)
    N, K = boxes.shape[0], query_boxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=np.float64)
    b = boxes[:, None, :]
    q = query_boxes[None, :, :]
    iw = np.minimum(b[..., 2], q[..., 2]) - np.maximum(b[..., 0], q[..., 0]) + 1
    ih = np.minimum(b[..., 3], q[..., 3]) - np.maximum(b[..., 1], q[..., 1]) + 1
    q_area = (q[..., 2] - q[..., 0] + 1) * (q[..., 3] - q[..., 1] + 1)
    b_area = (b[..., 2] - b[..., 0] + 1) * (b[..., 3] - b[..., 1] + 1)
    inter = iw * ih
    valid = (iw > 0) & (ih > 0)
    ua = b_area + q_area - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(valid, inter / ua, 0.0)
    return out
