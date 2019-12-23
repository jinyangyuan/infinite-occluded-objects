import argparse
import numpy as np
import os
from common import load_objects, create_dataset


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


def add_color(objects_prev, colors, colors_compatible, only_object, same_color):
    objects = np.empty((objects_prev.shape[0], colors[0].shape[0] + 1, *objects_prev.shape[2:]),
                       dtype=objects_prev.dtype)
    objects[:, -1] = objects_prev[:, -1]
    idx_back = 0 if only_object else np.random.randint(len(colors))
    color_back = colors[idx_back]
    colors_candidate = colors_compatible[idx_back]
    if same_color:
        color_obj = colors_candidate[np.random.randint(len(colors_candidate))]
        objects[1:, :-1] = color_obj[None] * objects_prev[1:, :-1]
    else:
        colors_obj = np.stack(
            [colors_candidate[n] for n in np.random.randint(len(colors_candidate), size=objects_prev.shape[0])])
        objects[1:, :-1] = colors_obj[:-1] * objects_prev[1:, :-1]
    objects[0, :-1] = color_back * objects_prev[0, -1:]
    return objects


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--folder_inputs')
    parser.add_argument('--folder_outputs')
    parser.add_argument('--only_object', type=int)
    parser.add_argument('--same_color', type=int)
    parser.add_argument('--color_divs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=265076)
    args = parser.parse_args()
    if not os.path.exists(args.folder_outputs):
        os.mkdir(args.folder_outputs)
    # Objects
    colors = [convert_color(idx, args.color_divs) for idx in range(pow(args.color_divs, 3))]
    colors_compatible = [compute_colors_compatible(color_ref, colors) for color_ref in colors]
    objects_prev = load_objects(args.folder_inputs, args.name)
    objects = {key: np.empty((*val.shape[:2], colors[0].shape[0] + 1, *val.shape[3:]), val.dtype)
               for key, val in objects_prev.items()}
    np.random.seed(args.seed)
    for key, val in objects_prev.items():
        for idx, val_sub in enumerate(val):
            objects[key][idx] = add_color(val_sub, colors, colors_compatible, args.only_object, args.same_color)
    create_dataset(os.path.join(args.folder_outputs, args.name), objects)
