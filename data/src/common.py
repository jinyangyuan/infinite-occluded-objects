import h5py
import numpy as np
import os


def generate_objects_sub(args, elements, num_objects, max_tries=10):
    objects = np.zeros((num_objects + 1, *elements['back'].shape), dtype=np.float32)
    bboxs = np.ones(elements['back'].shape[1:], dtype=np.bool)
    while True:
        objects.fill(0)
        objects[0, -1] = 1
        bboxs.fill(True)
        sub_elements = []
        indices = np.random.randint(len(elements['objects']), size=num_objects)
        for idx in indices:
            sub_elements.append(elements['objects'][idx])
        for idx in range(1, objects.shape[0]):
            image = sub_elements[idx - 1]
            valid = False
            for _ in range(max_tries):
                row1 = np.random.randint(elements['back'].shape[-2] - image.shape[-2] + 1)
                row2 = row1 + image.shape[-2]
                col1 = np.random.randint(elements['back'].shape[-1] - image.shape[-1] + 1)
                col2 = col1 + image.shape[-1]
                if args.occlusion or np.all(bboxs[row1:row2, col1:col2]):
                    objects[idx, :, row1:row2, col1:col2] = image
                    bboxs[row1:row2, col1:col2] = False
                    valid = True
                    break
            if not valid:
                break
        else:
            break
    return objects


def generate_objects(args, elements):
    num_data = {'train': args.num_train, 'valid': args.num_valid, 'test': args.num_test}
    num_objects_all = np.array(args.num_objects_all)
    max_objects = num_objects_all.max()
    objects = {key: np.zeros((num_data[key], max_objects + 1, *elements[key]['back'].shape), dtype=np.float32)
               for key in num_data}
    np.random.seed(args.seed)
    for key, val in elements.items():
        num_objects_list = np.random.choice(num_objects_all, size=num_data[key])
        for idx, num_objects in enumerate(num_objects_list):
            objects[key][idx, :num_objects + 1] = generate_objects_sub(args, val, num_objects)
    return objects


def generate_images(objects):
    images = objects[:, 0, :-1]
    for idx in range(1, objects.shape[1]):
        masks = objects[:, idx, -1:]
        images = images * (1 - masks) + objects[:, idx, :-1] * masks
    return images


def generate_labels_mask(objects, th=0.5):
    masks_rev = objects[:, ::-1, -1]
    part_cumprod = np.concatenate([
        np.ones((masks_rev.shape[0], 1, *masks_rev.shape[2:]), dtype=masks_rev.dtype),
        np.cumprod(1 - masks_rev[:, :-1], 1),
    ], axis=1)
    coef = (masks_rev * part_cumprod)[:, ::-1]
    values = np.argmax(coef, 1).astype(np.int32)
    counts = (masks_rev >= th).sum(1) - 1
    valids = (counts == 1).astype(values.dtype)
    labels = np.stack([values, valids], axis=1)
    return labels


def generate_labels_rgba(objects):
    labels = objects.copy()
    masks = labels[:, 1:, -1:]
    labels[:, 1:, :-1] = labels[:, 1:, :-1] * masks + labels[:, :1, :-1] * (1 - masks)
    return labels


def create_dataset(name, objects):
    images = {key: generate_images(val) for key, val in objects.items()}
    labels_mask = {key: generate_labels_mask(val) for key, val in objects.items()}
    labels_rgba = {key: generate_labels_rgba(val) for key, val in objects.items()}
    with h5py.File('{}_data.h5'.format(name), 'w') as f:
        for key, val in images.items():
            f.create_dataset(key, data=val, compression='gzip', chunks=True)
    with h5py.File('{}_labels.h5'.format(name), 'w') as f:
        f.create_group('mask')
        for key, val in labels_mask.items():
            f['mask'].create_dataset(key, data=val, compression='gzip', chunks=True)
        f.create_group('rgba')
        for key, val in labels_rgba.items():
            f['rgba'].create_dataset(key, data=val, compression='gzip', chunks=True)
    return


def load_objects(folder, name):
    with h5py.File(os.path.join(folder, '{}_labels.h5'.format(name)), 'r') as f:
        labels_rgba = {key: f['rgba'][key][()] for key in f['rgba']}
    objects = {}
    for key, val in labels_rgba.items():
        masks = val[:, 1:, -1:].repeat(val.shape[2] - 1, axis=2)
        pos_sel = masks != 0
        val[:, 1:, :-1][pos_sel] = (val[:, 1:, :-1] - val[:, :1, :-1] * (1 - masks))[pos_sel] / masks[pos_sel]
        objects[key] = val
    return objects
