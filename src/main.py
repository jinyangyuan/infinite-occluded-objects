import os
from argument import get_argument
from dataset import get_dataloader
from model import get_model
from run_model import train_model, test_model


if __name__ == '__main__':
    args = get_argument()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataloader = get_dataloader(args)
    if args.train:
        if not os.path.exists(args.folder):
            os.mkdir(args.folder)
        net = get_model(args).cuda()
        train_model(args, net, dataloader)
    else:
        net = get_model(args, os.path.join(args.folder, args.file_model)).cuda()
        test_model(args, net, dataloader)
