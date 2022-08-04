import os
osp = os.path
import numpy as np

import args as args_module
from data.dataloader import get_dataloaders
from models.stargan import train_model as train_starGAN
from models.caae import train_model as train_CAAE
from models.rgae import train_model as train_RGAE
from models.ipgan import train_model as train_IPGAN


def main(seed, args):
    import torch
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")

    if seed >= 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        args["random seed"] = seed

    dataloaders = get_dataloaders(args)
    if args["network"]["type"] == "StarGAN":
        train_starGAN(args, dataloaders=dataloaders)
    elif args["network"]["type"] == "RGAE":
        train_RGAE(args, dataloaders=dataloaders)
    elif args["network"]["type"] == "CAAE":
        train_CAAE(args, dataloaders=dataloaders)
    elif args["network"]["type"] == "IPGAN":
        train_IPGAN(args, dataloaders=dataloaders)


if __name__ == "__main__":
    args = args_module.parse_args()
    main(args["random seed"], args)
