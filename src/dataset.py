import h5py
import torch
import torch.utils.data as utils_data


class Dataset(utils_data.Dataset):

    def __init__(self, data):
        super(Dataset, self).__init__()
        self.images = torch.tensor(data['image'])
        self.data_slots = data['segment'].max() + 1
        self.segments = torch.tensor(data['segment'].max() - data['segment'])
        self.overlaps = torch.tensor(data['overlap'] > 1)
        self.layers = torch.tensor(data['layers'][:, ::-1].copy()) if 'layers' in data else None

    def __getitem__(self, idx):
        image = self.images[idx].float() / 255
        segment = self.segments[idx].long().unsqueeze_(0)
        segment = torch.zeros([self.data_slots, *segment.shape[1:]]).scatter_(0, segment, 1).unsqueeze_(-3)
        overlap = self.overlaps[idx].float().unsqueeze_(-3)
        data = {'image': image, 'segment': segment, 'overlap': overlap}
        if self.layers is not None:
            data['layers'] = self.layers[idx].float() / 255
        return data

    def __len__(self):
        return self.images.shape[0]


def get_data_loaders(config):
    image_shape = None
    data_loaders = {}
    with h5py.File(config['path_data'], 'r', libver='latest', swmr=True) as f:
        phase_list = [*f.keys()]
        if not config['train']:
            phase_list = [n for n in phase_list if n not in ['train', 'valid']]
        for phase in phase_list:
            data = {key: f[phase][key][()] for key in ['image', 'segment', 'overlap']}
            if 'layers' in f[phase] and phase not in ['train', 'valid']:
                data['layers'] = f[phase]['layers'][()]
            if image_shape is None:
                image_shape = data['image'].shape[-3:]
            else:
                assert image_shape == data['image'].shape[-3:]
            data_loaders[phase] = utils_data.DataLoader(
                Dataset(data),
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                shuffle=(phase == 'train'),
                drop_last=(phase == 'train'),
                pin_memory=True,
            )
    return data_loaders, image_shape
