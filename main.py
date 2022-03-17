import argparse
import matplotlib
import os
import yaml
from train import train


# environment
matplotlib.use('agg')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()

    config = open(args.config)
    config = yaml.load(config, Loader=yaml.FullLoader)
    train_config = config['train_config']
    train(args.resume, args.ckpt_path, **train_config)


if __name__ == "__main__":
    main()
