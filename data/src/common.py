import h5py
import numpy as np
import os


def create_dataset(name, images, labels_ami, labels_mse):
    with h5py.File('{}_data.h5'.format(name), 'w') as f:
        f.create_group('Image')
        for key, val in images.items():
            f['Image'].create_dataset(key, data=val, compression='gzip')
        f.create_group('Background')
        for key, val in labels_mse.items():
            f['Background'].create_dataset(key, data=val[:, -1], compression='gzip')
    with h5py.File('{}_labels.h5'.format(name), 'w') as f:
        f.create_group('AMI')
        for key, val in labels_ami.items():
            f['AMI'].create_dataset(key, data=val, compression='gzip')
        f.create_group('MSE')
        for key, val in labels_mse.items():
            f['MSE'].create_dataset(key, data=val, compression='gzip')


def load_dataset(folder, name):
    with h5py.File(os.path.join(folder, '{}_data.h5'.format(name)), 'r') as f:
        images = {key: f['Image'][key][()] for key in f['Image']}
    with h5py.File(os.path.join(folder, '{}_labels.h5'.format(name)), 'r') as f:
        labels_ami = {key: f['AMI'][key][()] for key in f['AMI']}
        labels_mse = {key: f['MSE'][key][()] for key in f['MSE']}
    return images, labels_ami, labels_mse


def generate_labels_occ(images_all, num_objects, image_size, threshold=0.5):
    indices = np.random.randint(len(images_all), size=num_objects)
    overlap_cnt = np.zeros((image_size, image_size), dtype=np.int)
    label_ami = np.zeros((image_size, image_size), dtype=np.float32)
    label_mse = np.zeros((num_objects, 1, image_size, image_size), dtype=np.float32)
    for idx_obj, idx_image in enumerate(indices):
        element = images_all[idx_image]
        row1 = np.random.randint(image_size - element.shape[0] + 1)
        row2 = row1 + element.shape[0]
        col1 = np.random.randint(image_size - element.shape[1] + 1)
        col2 = col1 + element.shape[1]
        pos_valid = element > threshold
        overlap_cnt[row1:row2, col1:col2][pos_valid] += 1
        label_ami[row1:row2, col1:col2][pos_valid] = idx_obj + 1
        label_mse[idx_obj, 0, row1:row2, col1:col2] = element
    label_ami[overlap_cnt != 1] = 0
    return label_ami, label_mse


def generate_labels_sep(images_all, num_objects, image_size, threshold=0.5, max_tries=10):
    while True:
        mark_canvas = True
        indices = np.random.randint(len(images_all), size=num_objects)
        valids = np.ones((image_size, image_size), dtype=np.bool)
        label_ami = np.zeros((image_size, image_size), dtype=np.float32)
        label_mse = np.zeros((num_objects, 1, image_size, image_size), dtype=np.float32)
        for idx_obj, idx_image in enumerate(indices):
            mark_element = False
            element = images_all[idx_image]
            for _ in range(max_tries):
                row1 = np.random.randint(image_size - element.shape[0] + 1)
                row2 = row1 + element.shape[0]
                col1 = np.random.randint(image_size - element.shape[1] + 1)
                col2 = col1 + element.shape[1]
                if np.all(valids[row1:row2, col1:col2]):
                    mark_element = True
                    valids[row1:row2, col1:col2] = False
                    label_ami[row1:row2, col1:col2][element > threshold] = idx_obj + 1
                    label_mse[idx_obj, 0, row1:row2, col1:col2] = element
                    break
            if not mark_element:
                mark_canvas = False
                break
        if mark_canvas:
            break
    return label_ami, label_mse


def generate_dataset(args, data):
    num_data_dict = {'train': args.num_train, 'valid': args.num_valid, 'test': args.num_test}
    num_objects_all = np.array(args.num_objects_all)
    max_objects = num_objects_all.max() + 1
    labels_ami = {key: np.empty((val, args.image_size, args.image_size), dtype=np.float32)
                  for key, val in num_data_dict.items()}
    labels_mse = {key: np.zeros((val, max_objects, 1, args.image_size, args.image_size), dtype=np.float32)
                  for key, val in num_data_dict.items()}
    np.random.seed(args.seed)
    for key, val in data.items():
        num_objects_list = np.random.choice(num_objects_all, size=num_data_dict[key])
        for idx, num_objects in enumerate(num_objects_list):
            if args.occlusion:
                labels_ami[key][idx], labels_mse[key][idx, :num_objects] = \
                    generate_labels_occ(val, num_objects, args.image_size)
            else:
                labels_ami[key][idx], labels_mse[key][idx, :num_objects] = \
                    generate_labels_sep(val, num_objects, args.image_size)
    images = {key: val.max(1) for key, val in labels_mse.items()}
    return images, labels_ami, labels_mse
