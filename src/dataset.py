import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = torch.tensor(data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def get_dataloaders(args):
    key_list = ['train', 'valid'] if args.train else ['test']
    with h5py.File(args.path_data, 'r', libver='latest', swmr=True) as f:
        data = {key: f[key][()] for key in key_list}
    image_planes, image_full_height, image_full_width = data[key_list[0]].shape[-3:]
    datasets = {key: CustomDataset(val) for key, val in data.items()}
    dataloaders = {
        key: DataLoader(
            val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=key == 'train',
            drop_last=key == 'train',
        )
        for key, val in datasets.items()
    }
    return dataloaders, image_planes, image_full_height, image_full_width
