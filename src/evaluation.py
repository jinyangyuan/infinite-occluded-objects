import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import hsv_to_rgb
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score


def load_dataset(folder, name):
    with h5py.File(os.path.join(folder, '{}_data.h5'.format(name)), 'r') as f:
        images = f['Image']['test'][()]
    with h5py.File(os.path.join(folder, '{}_labels.h5'.format(name)), 'r') as f:
        labels_ami = f['AMI']['test'][()]
        labels_mse = f['MSE']['test'][()]
    return images, labels_ami, labels_mse


def load_results(folder, filename='result.h5'):
    with h5py.File(os.path.join(folder, filename), 'r') as f:
        results = {key: f[key][()] for key in f}
    return results


def convert_results(results):
    gamma = results['gamma']
    pres = results['pres'].squeeze(2)
    shape = results['shape']
    appear_combine = results['appear_combine']
    coef = pres[..., None, None, None] * shape
    recon_scene = (gamma * appear_combine).sum(1)
    recon_objects = coef * appear_combine[:, :-1] + (1 - coef) * appear_combine[:, -1:]
    return gamma, pres, coef, recon_scene, recon_objects


def compute_ami_score(gamma, labels_ami):
    predictions = gamma[:, :-1].argmax(1).squeeze(1)
    scores = []
    for prediction, label in zip(predictions, labels_ami):
        pos_sel = np.where(label != 0)
        scores.append(adjusted_mutual_info_score(prediction[pos_sel], label[pos_sel]))
    return np.mean(scores)


def compute_mse_score(recon_objects, labels_mse):
    scores = []
    for recon, label in zip(recon_objects, labels_mse):
        cost = np.ndarray((label.shape[0], recon.shape[0]))
        for i in range(label.shape[0]):
            for j in range(recon.shape[0]):
                cost[i, j] = np.square(label[i] - recon[j]).mean()
        rows, cols = linear_sum_assignment(cost)
        scores.append(cost[rows, cols])
    return np.mean(scores)


def compute_oca_score(pres, labels_ami):
    scores = (pres > 0.5).sum(1) == labels_ami.reshape(labels_ami.shape[0], -1).max(1)
    return np.mean(scores)


def compute_orders(recon_objects, labels_mse):
    orders = []
    for recon, label in zip(recon_objects, labels_mse):
        cost = np.ndarray((label.shape[0], recon.shape[0]))
        for i in range(label.shape[0]):
            for j in range(recon.shape[0]):
                cost[i, j] = np.square(label[i] - recon[j]).mean()
        _, cols = linear_sum_assignment(cost)
        orders.append(cols)
    return np.array(orders)


def compute_weights(labels_mse):
    masks = np.sum(np.abs(labels_mse - labels_mse[:, -1:]), axis=-3, keepdims=True) != 0
    weights = np.zeros((masks.shape[0], masks.shape[1], masks.shape[1]))
    for i in range(masks.shape[1] - 1):
        for j in range(i + 1, masks.shape[1]):
            diff_sq = np.square(labels_mse[:, i] - labels_mse[:, j]).sum(-3, keepdims=True) * masks[:, i] * masks[:, j]
            weights[:, i, j] = diff_sq.reshape(diff_sq.shape[0], -1).sum(-1)
    return weights


def adjust_recon_objects(recon_objects, images, coef):
    diffs_sq = np.square(recon_objects - images[:, None]).sum(-3, keepdims=True)
    recon_objects_adj = []
    for sub_recon_objects, sub_images, sub_coef, sub_diffs_sq in zip(recon_objects, images, coef, diffs_sq):
        cand_list = list(range(sub_recon_objects.shape[0]))
        perm_list = []
        for _ in range(sub_recon_objects.shape[0]):
            score_mat = sub_coef[cand_list] * sub_diffs_sq[cand_list]
            score = score_mat.reshape(score_mat.shape[0], -1).sum(-1)
            idx_sel = int(np.argmin(score))
            perm_list.append(cand_list[idx_sel])
            cand_list.pop(idx_sel)
            sub_diffs_sq = (1 - sub_coef[idx_sel])[None] * sub_diffs_sq
        recon_objects_adj.append(sub_recon_objects[perm_list])
    return np.stack(recon_objects_adj)


def compute_ooa_score(recon_objects, labels_mse, weights):
    orders = compute_orders(recon_objects, labels_mse)
    binary_mat = np.zeros(weights.shape)
    for i in range(orders.shape[1] - 1):
        for j in range(i + 1, orders.shape[1]):
            binary_mat[:, i, j] = orders[:, i] > orders[:, j]
    return (binary_mat * weights).sum() / weights.sum()


def compute_scores(images, labels_ami, labels_mse, gamma, pres, coef, recon_objects, is_ordered):
    ami_score = compute_ami_score(gamma, labels_ami)
    mse_score = compute_mse_score(recon_objects, labels_mse)
    oca_score = compute_oca_score(pres, labels_ami)
    ooa_score_ori, ooa_score_adj = -1, -1
    if is_ordered:
        weights = compute_weights(labels_mse)
        recon_objects_adj = adjust_recon_objects(recon_objects, images, coef)
        ooa_score_ori = compute_ooa_score(recon_objects, labels_mse, weights)
        ooa_score_adj = compute_ooa_score(recon_objects_adj, labels_mse, weights)
    return ami_score, mse_score, oca_score, ooa_score_ori, ooa_score_adj


def compute_gamma_combine(gamma, num_colors):
    hsv_colors = np.ones((num_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, num_colors, endpoint=False) + 2 / 3) % 1.0
    gamma_colors = hsv_to_rgb(hsv_colors)
    gamma_combine = np.clip((gamma * gamma_colors[None, ..., None, None]).sum(1), 0, 1)
    return gamma_combine, gamma_colors


def convert_image(image):
    image = (np.transpose(np.clip(image, 0, 1), [1, 2, 0]) * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def color_spines(ax, color, lw=3):
    for loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[loc].set_linewidth(lw)
        ax.spines[loc].set_color(color)
        ax.spines[loc].set_visible(True)


def plot_image(ax, image, xlabel=None, ylabel=None, border_color=None):
    ax.imshow(image, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color='k') if xlabel else None
    ax.set_ylabel(ylabel, color='k') if ylabel else None
    ax.xaxis.set_label_position('top')
    if border_color:
        color_spines(ax, color=border_color)


def plot_samples(images, gamma, recon_scene, recon_objects, num_images=15):
    gamma_combine, gamma_colors = compute_gamma_combine(gamma, gamma.shape[1])
    nrows, ncols = recon_objects.shape[1] + 3, num_images
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    for idx in range(num_images):
        plot_image(axes[0, idx], convert_image(images[idx]), ylabel='scene' if idx == 0 else None)
        plot_image(axes[1, idx], convert_image(recon_scene[idx]), ylabel='recon' if idx == 0 else None)
        for idx_sub in range(recon_objects.shape[1]):
            plot_image(axes[idx_sub + 2, idx], convert_image(recon_objects[idx, idx_sub]),
                       ylabel='obj {}'.format(idx_sub + 1) if idx == 0 else None,
                       border_color=tuple(gamma_colors[idx_sub]))
        plot_image(axes[-1, idx], convert_image(gamma_combine[idx]), ylabel='segre' if idx == 0 else None)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
