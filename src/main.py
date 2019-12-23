import os
import yaml
from argument import get_arguments
from dataset import get_dataloaders
from model import get_model
from run_model import train_model, test_model


if __name__ == '__main__':
    args = get_arguments()
    dataloaders, args.image_planes, args.image_full_height, args.image_full_width = get_dataloaders(args)
    args.image_crop_height = args.image_full_height // 2
    args.image_crop_width = args.image_full_width // 2
    if args.train:
        if not os.path.exists(args.folder):
            os.mkdir(args.folder)
        exclude_keys = [
            'path_config', 'folder', 'train',
            'file_args', 'file_log', 'file_model', 'file_result_base',
        ]
        args_dict = {key: val for key, val in args.__dict__.items() if key not in exclude_keys}
        with open(os.path.join(args.folder, args.file_args), 'w') as f:
            yaml.safe_dump(args_dict, f)
        net = get_model(args).cuda()
        train_model(args, net, dataloaders)
    else:
        net = get_model(args, os.path.join(args.folder, args.file_model)).cuda()
        test_model(args, net, dataloaders)
