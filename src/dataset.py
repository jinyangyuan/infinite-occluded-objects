import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):

    def __init__(self, data):
        self.images = data['image']
        self.backgrounds = data['background']

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx])
        background = torch.FloatTensor(self.backgrounds[idx])
        return image, background

    def __len__(self):
        return self.images.shape[0]


def get_dataloader(args):
    with h5py.File(args.path_data, 'r', libver='latest', swmr=True) as f:
        data = {
            key: {
                'image': f['Image'][key][()],
                'background': f['Background'][key][()],
            }
            for key in (['train', 'valid'] if args.train else ['test'])
        }
    dataset = {key: CustomDataset(data[key]) for key in data}
    dataloader = {
        key: DataLoader(
            val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=key == 'train',
            drop_last=key == 'train'
        )
        for key, val in dataset.items()
    }
    return dataloader
