import os, yaml, torch
osp = os.path

from data import dataloader
from models.stargan import build_starGAN
from models.rgae import build_RGAE
from models.caae import build_CAAE
from models.ipgan import build_IPGAN

def get_job_args(config_path):
    args = yaml.safe_load(open(config_path, "r"))
    return args

def get_job_model_and_args(config_path, results_dir):
    args = get_job_args(config_path)
    G_path = osp.join(results_dir, "weights/final_G.pth")
    if osp.exists(G_path):
        prefix = "final"
    else:
        prefix = "best"
        G_path = osp.join(results_dir, "weights/best_G.pth")

    if args["network"]["type"] == "StarGAN":
        G,DR = build_starGAN(args)
        DR_path = osp.join(results_dir, "weights/{prefix}_DR.pth")
        DR.load_state_dict(torch.load(DR_path), strict=False)
        models = {"G":G, "DR":DR}
    elif args["network"]["type"] == "CAAE":
        G,Dz,Dimg = build_CAAE(args)
        Dz_path = osp.join(results_dir, "weights/{prefix}_Dz.pth")
        Dimg_path = osp.join(results_dir, "weights/{prefix}_Dimg.pth")
        Dz.load_state_dict(torch.load(Dz_path), strict=False)
        Dimg.load_state_dict(torch.load(Dimg_path), strict=False)
        models = {"G":G, "Dz":Dz, "Dimg":Dimg}
    elif args["network"]["type"] == "RGAE":
        G,R = build_RGAE(args)
        R_path = osp.join(results_dir, "weights/{prefix}_R.pth")
        R.load_state_dict(torch.load(R_path), strict=False)
        models = {"G":G, "R":R}
    elif args["network"]["type"] == "IPGAN":
        G,D = build_IPGAN(args)
        D_path = osp.join(results_dir, "weights/{prefix}_D.pth")
        D.load_state_dict(torch.load(D_path), strict=False)
        models = {"G":G, "D":D}
    else:
        raise NotImplementedError

    G.load_state_dict(torch.load(G_path), strict=False)
    for m, model in models.items():
        model.eval()
    return models, args


def build_synthetic_dataset_for_job(config_path):
    models, args = get_job_model_and_args(config_path)
    dataloaders = dataloader.get_dataloaders_for_dataset(args["dataset"], batch_size=32, attr_loaders=True)
    G = models["G"]
    fake_imgs, attrs_new = [], []
    with torch.no_grad():
        attr_iter = dataloaders["val_attr"].__iter__()
        for batch in dataloaders["val"]:
            orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
            attr_new = next(attr_iter).cuda()
            attr_new = torch.where(torch.isnan(attr_new), torch.randn_like(attr_new), attr_new)
            attr_gt = torch.where(torch.isnan(attr_gt), torch.zeros_like(attr_gt), attr_gt)
            dY = attr_new - attr_gt
            fake_img = G(orig_imgs, dY)
            fake_imgs.append(fake_img.cpu())
            attrs_new.append(attr_new.cpu())

    return torch.cat(fake_imgs,0), torch.cat(attrs_new,0)
