import argparse
import yaml


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--path_data')
    parser.add_argument('--folder')
    parser.add_argument('--file_result_base')
    parser.add_argument('--train', type=int)
    parser.add_argument('--max_objects', type=int)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--num_tests', type=int)
    args = parser.parse_args()
    with open(args.path_config) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key != 'path_config' and val is not None:
            config[key] = val
    args.__dict__ = config
    if args.loss_weights is None:
        args.loss_weights = [1.] * (args.num_steps + 1)
    assert len(args.loss_weights) == args.num_steps + 1
    return args
