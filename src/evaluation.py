import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2gray
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


def load_dataset(folder, name, phase='test'):
    with h5py.File(os.path.join(folder, '{}_data.h5'.format(name)), 'r') as f:
        images = f[phase][()]
    with h5py.File(os.path.join(folder, '{}_labels.h5'.format(name)), 'r') as f:
        labels = {key: f[key][phase][()] for key in f}
    return images, labels


def load_results(folder, filename='result_0.h5'):
    with h5py.File(os.path.join(folder, filename), 'r') as f:
        results = {key: f[key][()] for key in f}
    return results


def compute_ll_score(log_likelihood):
    return np.mean(log_likelihood)


def compute_segre_score(segre, labels_mask, exclude_back):
    if exclude_back:
        predictions = segre[:, :-1].argmax(1).squeeze(1)
    else:
        predictions = segre.argmax(1).squeeze(1)
    ami_scores, ari_scores = [], []
    for prediction, label in zip(predictions, labels_mask):
        pos_sel = np.where(label[-1] != 0)
        ami_scores.append(adjusted_mutual_info_score(label[0][pos_sel], prediction[pos_sel]))
        ari_scores.append(adjusted_rand_score(label[0][pos_sel], prediction[pos_sel]))
    return np.mean(ami_scores), np.mean(ari_scores)


def compute_rmse_score(orders, recon_objects, labels_rgba):
    objects_rgb, objects_a = labels_rgba[:, 1:, :-1], labels_rgba[:, 1:, -1]
    recon_objects_sel = np.array([n[order] for n, order in zip(recon_objects, orders)])
    diffs_sq = np.square(objects_rgb - recon_objects_sel).mean(-3)
    return np.sqrt((diffs_sq * objects_a).sum() / objects_a.sum())


def compute_oca_score(pres, labels_mask):
    num_objects = labels_mask[:, 0].reshape(labels_mask.shape[0], -1).max(1)
    return np.mean(pres.sum(1) == num_objects)


def compute_ooa_score(orders, labels_rgba):
    objects_rgb, objects_a = labels_rgba[:, 1:, :-1], labels_rgba[:, 1:, -1]
    weights = np.zeros((objects_a.shape[0], objects_a.shape[1], objects_a.shape[1]))
    for i in range(objects_a.shape[1] - 1):
        for j in range(i + 1, objects_a.shape[1]):
            diffs_sq = np.square(objects_rgb[:, i] - objects_rgb[:, j]).sum(-3)
            diffs_sq *= objects_a[:, i] * objects_a[:, j]
            weights[:, i, j] = diffs_sq.reshape(diffs_sq.shape[0], -1).sum(-1)
    binary_mat = np.zeros(weights.shape)
    for i in range(orders.shape[1] - 1):
        for j in range(i + 1, orders.shape[1]):
            binary_mat[:, i, j] = orders[:, i] > orders[:, j]
    return (binary_mat * weights).sum() / weights.sum()


def compute_orders(segre, labels_mask, labels_rgba):
    values = segre[:, :-1]
    orders = np.zeros((labels_rgba.shape[0], labels_rgba.shape[1] - 1), dtype=np.int)
    for idx, (value, label) in enumerate(zip(values, labels_mask[:, :-1])):
        num_objects = label.max()
        cost = np.ndarray((num_objects, value.shape[0]))
        for i in range(num_objects):
            pos = np.where(label == i + 1)
            for j in range(value.shape[0]):
                cost[i, j] = -value[j][pos].sum()
        _, cols = linear_sum_assignment(cost)
        orders[idx, :cols.size] = cols
    return orders


def compute_scores(results, labels, is_ordered=True, exclude_segre_back=True):
    scores = {}
    orders = compute_orders(results['segre'], labels['mask'], labels['rgba'])
    scores['LL_M'] = compute_ll_score(results['ll_mixture'])
    scores['LL_S'] = compute_ll_score(results['ll_single'])
    scores['AMI'], scores['ARI'] = compute_segre_score(results['segre'], labels['mask'], exclude_segre_back)
    scores['RMSE'] = compute_rmse_score(orders, results['recon_objects'], labels['rgba'])
    scores['OCA'] = compute_oca_score(results['pres'], labels['mask'])
    if is_ordered:
        scores['OOA'] = compute_ooa_score(orders, labels['rgba'])
    return scores


def compute_segre_combine(segre):
    num_colors = segre.shape[1]
    hsv_colors = np.ones((num_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, num_colors, endpoint=False) + 2 / 3) % 1.0
    segre_colors = hsv_to_rgb(hsv_colors)
    segre_combine = np.clip((segre * segre_colors[None, ..., None, None]).sum(1), 0, 1)
    return segre_combine, segre_colors


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
    return


def plot_image(ax, image, xlabel=None, ylabel=None, border_color=None):
    ax.imshow(image, interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color='k') if xlabel else None
    ax.set_ylabel(ylabel, color='k') if ylabel else None
    ax.xaxis.set_label_position('top')
    if border_color:
        color_spines(ax, color=border_color)
    return


def plot_samples(images, results, num_images=15, scale=1):
    segre_combine, segre_colors = compute_segre_combine(results['segre'])
    max_objects = results['recon_objects'].shape[1]
    nrows, ncols = max_objects + 4, num_images
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * scale, nrows * scale))
    for idx in range(num_images):
        plot_image(axes[0, idx], convert_image(images[idx]), ylabel='scene' if idx == 0 else None)
        plot_image(axes[1, idx], convert_image(results['recon_scene'][idx]), ylabel='recon' if idx == 0 else None)
        image_gray = rgb2gray(convert_image(images[idx]))
        bbox_color = 'w' if np.median(image_gray) < 0.25 else 'k'
        for idx_sub in range(max_objects):
            plot_image(axes[idx_sub + 2, idx], convert_image(results['recon_objects'][idx, idx_sub]),
                       ylabel='obj {}'.format(idx_sub + 1) if idx == 0 else None,
                       border_color=tuple(segre_colors[idx_sub]) if results['pres'][idx, idx_sub] else None)
            if results['pres'][idx, idx_sub]:
                image_shape = np.array(images.shape[-2:])
                center = (results['trs'][idx, idx_sub] + 1) * 0.5 * (image_shape - 1)
                shape = results['scl'][idx, idx_sub] * image_shape
                corner = center - shape * 0.5
                rect = Rectangle(corner, shape[0], shape[1], linewidth=1, edgecolor=bbox_color, facecolor='none')
                axes[idx_sub + 2, idx].add_patch(rect)
        plot_image(axes[-2, idx], convert_image(results['back'][idx]), ylabel='back' if idx == 0 else None,
                   border_color=tuple(segre_colors[-1]))
        plot_image(axes[-1, idx], convert_image(segre_combine[idx]), ylabel='segre' if idx == 0 else None)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig
