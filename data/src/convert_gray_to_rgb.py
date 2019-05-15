import argparse
import numpy as np
import os
from common import create_dataset, load_dataset


def convert_color(x, color_divs):
    color = np.empty(3, dtype=np.float32)
    for idx in range(color.shape[0]):
        color[idx] = x % color_divs
        x //= color_divs
    return color[:, None, None] / (color_divs - 1)


def compute_colors_compatible(color_ref, colors):
    compatible_list = []
    for color in colors:
        if np.square(color - color_ref).sum() >= 0.75:
            compatible_list.append(color)
    return compatible_list


def add_color_dep(image, label_mse, colors, colors_compatible, only_object):
    idx_back = 0 if only_object else np.random.randint(len(colors))
    color_back = colors[idx_back]
    colors_candidate = colors_compatible[idx_back]
    color_obj = colors_candidate[np.random.randint(len(colors_candidate))]
    image_new = image * color_obj + (1 - image) * color_back
    label_mse_new = label_mse * color_obj[None] + (1 - label_mse) * color_back[None]
    return image_new, label_mse_new


def add_color_ind(image, label_mse, colors, colors_compatible, only_object):
    idx_back = 0 if only_object else np.random.randint(len(colors))
    color_back = colors[idx_back]
    colors_candidate = colors_compatible[idx_back]
    colors_obj = np.stack(
        [colors_candidate[n] for n in np.random.randint(len(colors_candidate), size=label_mse.shape[0])])
    image_new = np.empty((3, *image.shape[1:]), dtype=image.dtype)
    image_new[...] = color_back
    for color_obj, mask in zip(colors_obj, label_mse):
        image_new = mask * color_obj + (1 - mask) * image_new
    label_mse_new = label_mse * colors_obj + (1 - label_mse) * color_back[None]
    return image_new, label_mse_new


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_inputs')
    parser.add_argument('--folder_outputs')
    parser.add_argument('--name')
    parser.add_argument('--dependence', type=int)
    parser.add_argument('--only_object', type=int)
    parser.add_argument('--num_objects_all', type=int, nargs='+')
    parser.add_argument('--color_divs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=265076)
    args = parser.parse_args()
    # Load previous dataset
    name = '{}_{}'.format(args.name, '_'.join([str(n) for n in args.num_objects_all]))
    images, labels_ami, labels_mse = load_dataset(args.folder_inputs, name)
    images_new = {key: np.empty((val.shape[0], 3, *val.shape[2:]), dtype=val.dtype) for key, val in images.items()}
    labels_mse_new = {key: np.empty((*val.shape[:2], 3, *val.shape[3:]), dtype=val.dtype)
                      for key, val in labels_mse.items()}
    # Create new dataset
    colors = [convert_color(idx, args.color_divs) for idx in range(pow(args.color_divs, 3))]
    colors_compatible = [compute_colors_compatible(color_ref, colors) for color_ref in colors]
    np.random.seed(args.seed)
    for key in labels_mse_new:
        for idx in range(labels_mse_new[key].shape[0]):
            if args.dependence:
                images_new[key][idx], labels_mse_new[key][idx] = add_color_dep(
                    images[key][idx], labels_mse[key][idx], colors, colors_compatible, args.only_object)
            else:
                images_new[key][idx], labels_mse_new[key][idx] = add_color_ind(
                    images[key][idx], labels_mse[key][idx], colors, colors_compatible, args.only_object)
    create_dataset(os.path.join(args.folder_outputs, name), images_new, labels_ami, labels_mse_new)


if __name__ == '__main__':
    main()
