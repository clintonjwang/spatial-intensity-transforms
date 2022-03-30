# import warnings
# warnings.simplefilter("ignore", UserWarning)

import os, yaml, torch, argparse, shutil
osp = os.path
from glob import glob
import dill as pickle
import numpy as np

import util, losses
from data.dataloader import get_dataloaders
from models.stargan import build_starGAN
from models.caae import build_CAAE
from models.cvae import build_CVAE
from models.ipgan import build_IPGAN
from analysis import analyze
from data import dataloader

ANALYSIS_DIR = osp.expanduser("~/code/sitgan/analysis")
RESULTS_DIR = osp.expanduser("~/code/sitgan/results")

def rename_job(job, new_name):
    os.rename(osp.join(RESULTS_DIR, job), osp.join(RESULTS_DIR, new_name))
    for folder in util.glob2(ANALYSIS_DIR, "*", job):
        os.rename(folder, folder.replace(job, new_name))

def delete_job(job):
    shutil.rmtree(osp.join(RESULTS_DIR, job))
    for folder in util.glob2(ANALYSIS_DIR, "*", job):
        shutil.rmtree(folder)

def get_job_args(job):
    config_path = osp.join(RESULTS_DIR, job, "config.yaml")
    args = yaml.safe_load(open(config_path, "r"))
    return args

def get_job_model_and_args(job):
    args = get_job_args(job)
    G_path = osp.expanduser(f"~/code/sitgan/results/{job}/weights/final_G.pth")
    if osp.exists(G_path):
        prefix = "final"
    else:
        prefix = "best"
        G_path = osp.expanduser(f"~/code/sitgan/results/{job}/weights/best_G.pth")

    if args["network"]["type"] == "StarGAN":
        G,DR = build_starGAN(args)
        DR_path = osp.expanduser(f"~/code/sitgan/results/{job}/weights/{prefix}_DR.pth")
        DR.load_state_dict(torch.load(DR_path), strict=False)
        models = {"G":G, "DR":DR}
    elif args["network"]["type"] == "CAAE":
        G,Dz,Dimg = build_CAAE(args)
        Dz_path = osp.expanduser(f"~/code/sitgan/results/{job}/weights/{prefix}_Dz.pth")
        Dimg_path = osp.expanduser(f"~/code/sitgan/results/{job}/weights/{prefix}_Dimg.pth")
        Dz.load_state_dict(torch.load(Dz_path), strict=False)
        Dimg.load_state_dict(torch.load(Dimg_path), strict=False)
        models = {"G":G, "Dz":Dz, "Dimg":Dimg}
    elif args["network"]["type"] == "CVAE":
        G = build_CVAE(args)
        models = {"G":G}
    elif args["network"]["type"] == "IPGAN":
        G,D = build_IPGAN(args)
        D_path = osp.expanduser(f"~/code/sitgan/results/{job}/weights/{prefix}_D.pth")
        D.load_state_dict(torch.load(D_path), strict=False)
        models = {"G":G, "D":D}
    else:
        raise NotImplementedError

    G.load_state_dict(torch.load(G_path), strict=False)
    for m, model in models.items():
        model.eval()
    return models, args

def get_dataset_for_job(job):
    return get_job_args(job)["dataset"]

def get_attributes_for_job(job):
    return get_job_args(job)["data loading"]["attributes"]

def get_repeated_jobs_from_base_job(base_job):
    job_names = []
    for results_folder in glob(f"{RESULTS_DIR}/{base_job}*"):
        job_names.append(osp.basename(results_folder)[:-4])
    if len(job_names) == 1:
        print("no repetitions/x-validation performed yet")
    return job_names

def get_synthetic_ds_for_job(job, behavior_if_missing="build"):
    path = osp.join(ANALYSIS_DIR, "synth_ds", job+"_imgs.pt")
    if not osp.exists(path):
        if behavior_if_missing == "build":
            build_synthetic_dataset_for_job(job)
        else:
            raise NotImplementedError
    imgs = torch.load(path)
    attrs = torch.load(path.replace("imgs", "attr"))
    return imgs, attrs

def build_synthetic_dataset_for_job(job, slurm=False):
    if slurm is True:
        return analyze.submit_job(build_synthetic_dataset_for_job, job, slurm=False, job_name=f"synth_{job}")

    dataset = get_dataset_for_job(job)
    dataloaders = dataloader.get_dataloaders_for_dataset(dataset, batch_size=32, attr_loaders=True)
    models, args = get_job_model_and_args(job)
    G = models["G"]
    model_type = args["network"]["type"]

    fake_imgs, attrs_new = [], []
    with torch.no_grad():
        attr_iter = dataloaders["val_attr"].__iter__()
        for batch in dataloaders["val"]:
            orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
            attr_new = next(attr_iter).cuda()
            attr_new = torch.where(torch.isnan(attr_new), torch.randn_like(attr_new), attr_new)
            attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
            dY = attr_new - attr_gt
            if model_type == "CVAE":
                fake_img = G(orig_imgs, y=attr_gt, dy=dY)
            elif model_type == "CAAE":
                fake_img = G(orig_imgs, y=attr_new)
            else:
                fake_img = G(orig_imgs, dY)
            fake_imgs.append(fake_img.cpu())
            attrs_new.append(attr_new.cpu())

    torch.save(torch.cat(fake_imgs,0), f=osp.join(ANALYSIS_DIR, "synth_ds", job+"_imgs.pt"))
    torch.save(torch.cat(attrs_new,0), f=osp.join(ANALYSIS_DIR, "synth_ds", job+"_attr.pt"))

def delete_synthetic_dataset_for_job(job, exact=True):
    if exact is False:
        for subjob in get_repeated_jobs_from_base_job(job):
            delete_synthetic_dataset_for_job(subjob, exact=True)
    else:
        path = osp.join(ANALYSIS_DIR, "synth_ds", job+"_imgs.pt")
        if osp.exists(path):
            os.remove(path)
            os.remove(path.replace("imgs", "attr"))






def get_sample_batch(job, phase="train", args=None):
    if args is None:
        args = util.get_job_args(job)
    dataloaders = get_dataloaders(args)
    for batch in dataloaders[phase]:
        return batch

def get_sample_outputs(job, phase="train"):
    G, DR, args = get_job_model_and_args(job)
    dataloaders = get_dataloaders(args)
    with torch.no_grad():
        for batch in dataloaders[phase]:
            orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
            fake_img, transforms = G(orig_imgs, dy, return_transforms=True)
            return {"orig_imgs": orig_imgs.cpu(), "fake_img":fake_img.cpu(), "transforms":transforms.cpu(),
                "attr_gt": attr_gt.cpu(), "dy": dy.cpu()}

def get_metrics_for_dataloader(models, dataloader, args, out_dir=None):
    for m in models:
        m.eval()
    network_settings = args["network"]
    pred_vols = []
    gt_vols = []
    dice_tracker = util.MetricTracker("test DICE")
    hausdorff_tracker = util.MetricTracker("test HD")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    metric_dict = {}
    with torch.no_grad():
        for batch in dataloader:
            imgs, gt_segs = batch["image"].cuda(), batch["label"]
            imgs = imgs.detach().squeeze(1).cpu().numpy()
            for i in range(pred_logits.shape[0]):
                seriesID = batch["seriesID"][i]

                pred_seg = (pred_logits[i] > 0).detach().squeeze().cpu().numpy()
                gt_seg = gt_segs[i].squeeze().numpy()

                dice = losses.dice_np(pred_seg, gt_seg)
                dice_tracker.update_with_minibatch(dice)
                hausd = skimage.metrics.hausdorff_distance(pred_seg, gt_seg)
                hausdorff_tracker.update_with_minibatch(hausd)

                metric_dict[seriesID] = (dice, hausd)

                if out_dir is not None:
                    root = osp.join(out_dir, seriesID)
                    util.save_example_slices(imgs[i], gt_seg, pred_seg, root=root)
                    with open(osp.join(root, "metrics.txt"), "w") as f:
                        f.write("%.2f\n%d" % (dice, hausd))


    if out_dir is not None:
        metrics_iter = sorted(metric_dict.items(), key=lambda item: item[1][0])
        with open(osp.join(out_dir, "metrics.txt"), "w") as f:
            f.write("Avg:\t%.2f\t%d\n" % (dice_tracker.epoch_average(), hausdorff_tracker.epoch_average()))
            for seriesID, metrics in metrics_iter:
                f.write("%s:\t%.2f\t%d\n" % (seriesID, metrics[0], metrics[1]))

    return dice_tracker, hausdorff_tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id')
    parser.add_argument('-s', '--slurm', action='store_true')
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()
    analyze.collect_metrics_for_jobs(args.job_id, exact=True, slurm=args.slurm, overwrite=args.overwrite)
