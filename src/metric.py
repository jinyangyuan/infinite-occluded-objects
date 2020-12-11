import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score


def compute_ami(seg_true_list, seg_pred_list, seg_valid_list):
    ami_list = []
    for seg_true, seg_pred, seg_valid in zip(seg_true_list, seg_pred_list, seg_valid_list):
        ami = adjusted_mutual_info_score(seg_true[seg_valid], seg_pred[seg_valid], average_method='max')
        ami_list.append(ami)
    return np.array(ami_list, dtype=np.float32)


def compute_order(cost_list):
    order_list = []
    for cost in cost_list:
        _, cols = linear_sum_assignment(cost)
        order_list.append(cols)
    return np.array(order_list)


def batch_flatten(x):
    return x.reshape(x.shape[0], -1)


def select_by_order(val_list, order_list):
    val_sel = np.array([val[order] for val, order in zip(val_list, order_list)])
    val_sel = np.concatenate([val_sel, val_list[:, -1:]], axis=1)
    return val_sel


def compute_ooa(layers_list, order_list):
    objects_apc, objects_shp = layers_list[:, :-1, :-1], layers_list[:, :-1, -1:]
    weights = np.zeros((objects_shp.shape[0], objects_shp.shape[1], objects_shp.shape[1]))
    for i in range(objects_shp.shape[1] - 1):
        for j in range(i + 1, objects_shp.shape[1]):
            sq_diffs = np.square(objects_apc[:, i] - objects_apc[:, j]).sum(-3, keepdims=True)
            sq_diffs *= objects_shp[:, i] * objects_shp[:, j]
            weights[:, i, j] = batch_flatten(sq_diffs).sum(-1)
    binary_mat = np.zeros(weights.shape)
    for i in range(order_list.shape[1] - 1):
        for j in range(i + 1, order_list.shape[1]):
            binary_mat[:, i, j] = order_list[:, i] < order_list[:, j]
    sum_scores = (binary_mat * weights).sum().astype(np.float32)
    sum_weights = weights.sum().astype(np.float32)
    return sum_scores, sum_weights


def compute_layer_mse(layers, apc, shp):
    target_noise = np.random.uniform(0, 1, size=apc.shape)
    noise = np.random.uniform(0, 1, size=apc.shape)
    target_apc, target_shp = layers[:, :, :-1], layers[:, :, -1:]
    target_recon = target_apc * target_shp + target_noise * (1 - target_shp)
    recon = apc * shp + noise * (1 - shp)
    sq_diffs = np.square(recon - target_recon).mean(-3, keepdims=True)
    mask_valid = 1 - (1 - target_shp) * (1 - shp)
    sum_scores = (sq_diffs * mask_valid).sum().astype(np.float32)
    sum_weights = mask_valid.sum().astype(np.float32)
    return sum_scores, sum_weights


def compute_iou_f1(seg_true_list, seg_pred_list, eps=1e-6):
    scores_iou_list, scores_f1_list, weights_list = [], [], []
    for seg_true, seg_pred in zip(seg_true_list, seg_pred_list):
        seg_true = batch_flatten(seg_true)
        seg_pred = batch_flatten(seg_pred)
        pres = (seg_true.max(-1) != 0).astype(np.float32)
        area_i = np.minimum(seg_true, seg_pred).sum(-1)
        area_u = np.maximum(seg_true, seg_pred).sum(-1)
        iou = area_i / (area_u + eps)
        f1 = 2 * area_i / (area_i + area_u + eps)
        scores_iou_list.append((iou * pres).sum())
        scores_f1_list.append((f1 * pres).sum())
        weights_list.append(pres.sum())
    sum_scores_iou = np.sum(scores_iou_list).astype(np.float32)
    sum_scores_f1 = np.sum(scores_f1_list).astype(np.float32)
    sum_weights = np.sum(weights_list).astype(np.float32)
    return (sum_scores_iou, sum_weights), (sum_scores_f1, sum_weights)
