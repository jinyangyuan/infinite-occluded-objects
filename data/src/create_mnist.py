import argparse
import numpy as np
import os
import torchvision
from common import create_dataset, generate_dataset


def convert_mnist_image(image):
    image = image[image.sum(1) != 0][:, image.sum(0) != 0] / 255
    return image.astype(np.float32)


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_downloads')
    parser.add_argument('--folder_outputs')
    parser.add_argument('--occlusion', type=int)
    parser.add_argument('--num_objects_all', type=int, nargs='+')
    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--num_train', type=int, default=50000)
    parser.add_argument('--num_valid', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=265076)
    args = parser.parse_args()
    # Convert MNIST
    mnist = {
        phase: torchvision.datasets.MNIST(
            args.folder_downloads, train=train, transform=None, target_transform=None, download=True)
        for phase, train in zip(['train', 'test'], [True, False])
    }
    mnist = {key: [convert_mnist_image(np.array(n[0])) for n in val] for key, val in mnist.items()}
    data = {key: mnist[key_mnist] for key, key_mnist in zip(['train', 'valid', 'test'], ['train', 'train', 'test'])}
    # Create dataset
    images, labels_ami, labels_mse = generate_dataset(args, data)
    name = 'mnist_{}'.format('_'.join([str(n) for n in args.num_objects_all]))
    create_dataset(os.path.join(args.folder_outputs, name), images, labels_ami, labels_mse)


if __name__ == '__main__':
    main()
