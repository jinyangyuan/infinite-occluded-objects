import h5py
import torch
import torch.utils.data as utils_data


class Dataset(utils_data.Dataset):

    def __init__(self, config, data):
        super(Dataset, self).__init__()
        self.images = torch.tensor(data['image'])
        self.segments = torch.tensor(data['segment'])
        self.overlaps = torch.tensor(data['overlap'])
        self.layers = torch.tensor(data['layers']) if 'layers' in data else None
        self.data_slots = self.segments.max() + 1
        self.seg_bck = config['seg_bck']
        self.seg_overlap = config['seg_overlap']

    def __getitem__(self, idx):
        image = self.images[idx].float() / 255
        segment = self.segments[idx].long().unsqueeze_(0)
        label = torch.zeros([self.data_slots, *segment.shape[1:]]).scatter_(0, segment, 1)
        if not self.seg_bck:
            label = label[1:]
        if not self.seg_overlap:
            valid = (self.overlaps[idx] <= 1).float().unsqueeze_(0)
            label *= valid
        label.unsqueeze_(-3)
        data = {'image': image, 'label': label}
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
                Dataset(config, data),
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                shuffle=(phase == 'train'),
                drop_last=(phase == 'train'),
                pin_memory=True,
            )
    return data_loaders, image_shape
