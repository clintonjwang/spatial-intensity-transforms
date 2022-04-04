import os, pdb, functools, submitit, shutil
osp = os.path

import numpy as np

import args as args_module
from data.dataloader import get_dataloaders
from models.stargan import train_model as train_starGAN
from models.caae import train_model as train_CAAE
from models.cvae import train_model as train_CVAE
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
    elif args["network"]["type"] == "CAAE":
        train_CAAE(args, dataloaders=dataloaders)
    elif args["network"]["type"] == "CVAE":
        train_CVAE(args, dataloaders=dataloaders)
    elif args["network"]["type"] == "IPGAN":
        train_IPGAN(args, dataloaders=dataloaders)


if __name__ == "__main__":
    args = args_module.parse_args()
    if "manual" in args["job_id"]:
        args["debug"] = True
        main(args["random seed"], args)
    else:
        executor = submitit.AutoExecutor(folder=args["paths"]["job output dir"])
        executor.update_parameters(
            name=args["job_id"],
            slurm_partition=args["partition"],
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=16,
            slurm_exclude="bergamot,perilla,caraway,cassia,anise,marjoram,clove,zaatar,mint", #clove
            slurm_exclusive=True if args["partition"]=="gpu" else False,
            timeout_min=90000,
        )
        if hasattr(args["random seed"], "__iter__"):
            fxn = functools.partial(main, args=args)
            jobs = executor.map_array(fxn, args["random seed"])
        else:
            job = executor.submit(main, args["random seed"], args)
