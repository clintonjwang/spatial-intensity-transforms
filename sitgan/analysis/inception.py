import os, torch, pdb
osp=os.path
nn=torch.nn
F=nn.functional
from glob import glob
import numpy as np
from tqdm import tqdm
import dill as pickle
import monai.transforms as mtr
import torchvision.models
from scipy import linalg

import util
from data import dataloader
import jobs as job_mgmt
from analysis import analyze 
from losses import maskedMSE_mean as masked_mse

ANALYSIS_DIR = osp.expanduser("~/code/sitgan/temp")

normalize_img = mtr.ScaleIntensityRangePercentiles(lower=2, upper=98, b_min=-1., b_max=1., clip=True)

def load_inception_regressor(dataset=None, tuned=True, activations_only=False):
    if tuned:
        if dataset == "2D ADNI T1":
            path = osp.join(ANALYSIS_DIR, "incv3/adni.pt")
        elif dataset == "2D MRI-GENIE FLAIR":
            path = osp.join(ANALYSIS_DIR, "incv3/mrigenie.pt")
        incv3 = torch.load(path).cuda().eval()
    else:
        os.environ["TORCH_HOME"] = ANALYSIS_DIR
        incv3 = torchvision.models.inception_v3(pretrained=True, progress=False, transform_input=False)
        input_layer = nn.Conv2d(1,32, kernel_size=3, stride=1, bias=False)
        input_layer.weight = nn.Parameter(incv3.Conv2d_1a_3x3.conv.weight[:,:1])
        incv3.Conv2d_1a_3x3.conv = input_layer
    if activations_only:
        incv3.fc = nn.Identity()
    return incv3.cuda().eval()

def fine_tune_regressor_on_dataset(dataset, slurm=False, save_path=None, bsz=32,
        n_iters=1000, lr=1e-4, val_freq=100):
    if slurm is True:
        return analyze.submit_job(fine_tune_regressor_on_dataset, dataset, job_name=f"{dataset[3:6]}_incv3",
                slurm=False, save_path=save_path, bsz=bsz, n_iters=n_iters, lr=lr, val_freq=val_freq)

    if dataset == "2D ADNI T1":
        outputs = ("age", "baseline diagnosis", "MMSE", "CDR")
        if save_path is None:
            save_path = "incv3/adni"
    elif dataset == "2D MRI-GENIE FLAIR":
        outputs = ("age", "NIHSS", "mRS")
        if save_path is None:
            save_path = "incv3/mrigenie"
    save_path = osp.join(ANALYSIS_DIR, save_path)

    incv3 = load_inception_regressor(tuned=False).train()
    incv3.fc = nn.Linear(2048, len(outputs), bias=True).cuda()

    dataloaders = dataloader.get_dataloaders_for_dataset(dataset, batch_size=bsz, augment=False)
    loss_tracker = util.MetricTracker("loss")
    trackers = [util.MetricTracker(var) for var in outputs]
    optimizer = torch.optim.Adam(incv3.parameters(), lr=lr, betas=(.5,.999))
    iter_num = 0
    while iter_num < n_iters:
        for batch in dataloaders["train"]:
            iter_num += 1
            Y_est = incv3(normalize_img(batch["image"].cuda()))[0]
            Y_gt = batch["attributes"].cuda()
            loss = masked_mse(Y_gt,Y_est)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tracker.update_with_minibatch(loss, phase="train")
            if iter_num % val_freq == 0:
                with torch.no_grad():
                    for batch in dataloaders["val"]:
                        incv3.eval()
                        Y_est = incv3(normalize_img(batch["image"].cuda()))
                        Y_gt = batch["attributes"].cuda()
                        loss = masked_mse(Y_gt,Y_est)
                        for ix in range(len(outputs)):
                            trackers[ix].update_with_minibatch(
                                masked_mse(Y_gt[...,ix], Y_est[...,ix]))
                        loss_tracker.update_with_minibatch(loss, phase="val")
                        incv3.train()
                    if loss_tracker.is_at_min("val"):
                        torch.save(incv3, save_path+".pt")
                        for ix,var in enumerate(outputs):
                            np.save(save_path+f"_{var}.npy", trackers[ix].minibatch_values["val"])
                    loss_tracker.update_at_epoch_end(phase="val")
                    for tracker in trackers:
                        tracker.update_at_epoch_end()

            if iter_num >= n_iters:
                break

    torch.save(incv3, save_path+"_end.pt")
    np.save(save_path+"_loss_train.npy", loss_tracker.get_moving_average(50, phase="train"))
    np.save(save_path+"_loss_val.npy", loss_tracker.epoch_history["val"])


########
# Regressor
########

def get_denormalization_factors_of_variable_for_dataset(variable, dataset):
    if dataset == "2D ADNI T1":
        norms = np.load(osp.join(ANALYSIS_DIR, "adni_normalizations.npy"))
        variables = ("age", "baseline diagnosis", "MMSE", "CDR")
    elif dataset == "2D MRI-GENIE FLAIR":
        norms = np.load(osp.join(ANALYSIS_DIR, "mrigenie_normalizations.npy"))
        variables = ("age", "NIHSS", "mRS")

    ix = variables.index(variable)
    return norms[ix]

# def load_age_regressor_residuals(job, overwrite=True):
#     path = osp.join(ANALYSIS_DIR, "residuals", job+"_residuals.npy")
#     if overwrite or not osp.exists(path):
#         compute_age_regressor_metrics_for_job(job, slurm=True)
#     return np.load(path)

def get_inception_v3_residuals(job, bsz=32, slurm=False, overwrite=False, tuned=True):
    if slurm is True:
        return analyze.submit_job(get_inception_v3_residuals, job, bsz=bsz, slurm=False,
            overwrite=overwrite, job_name=f"incv3_res_{job}")

    if tuned:
        path = osp.join(ANALYSIS_DIR, "residuals", f"{job}_t.dat")
    else:
        path = osp.join(ANALYSIS_DIR, "residuals", f"{job}.dat")
    if osp.exists(path) and overwrite is False:
        return pickle.load(open(path, "rb"))
        
    gen_imgs, Y_gt = job_mgmt.get_synthetic_ds_for_job(job, overwrite=overwrite)
    dataset = job_mgmt.get_dataset_for_job(job)
    incv3 = load_inception_regressor(dataset=dataset, tuned=tuned)
    outputs = job_mgmt.get_attributes_for_job(job)
    trackers = [util.MetricTracker(var) for var in outputs]

    def masked_diff(x,y):
        diffs = x-y
        return diffs[~torch.isnan(diffs)]

    with torch.no_grad():
        for B in range(0,len(gen_imgs),bsz):
            Y_est = incv3(normalize_img(gen_imgs[B:B+bsz].cuda())).cpu()
            for ix,tracker in enumerate(trackers):
                tracker.update_with_minibatch(masked_diff(Y_gt[B:B+bsz,ix], Y_est[:,ix]))

    residuals = {}
    for ix,var in enumerate(outputs):
        _,scale = get_denormalization_factors_of_variable_for_dataset(var, dataset=dataset)
        residuals[var] = np.array(trackers[ix].minibatch_values["val"]) * scale
    pickle.dump(residuals, open(path, "wb"))
    return residuals


########
# Image distributions
########

def get_incv3_activations_of_G(job, bsz=32, slurm=False, overwrite=False, tuned=True):
    if slurm is True:
        return analyze.submit_job(get_incv3_activations_of_G, job, bsz=bsz, slurm=False, tuned=tuned,
            overwrite=overwrite, job_name=f"incv3_act_{job}")

    if tuned:
        path = osp.join(ANALYSIS_DIR, "activations", f"{job}_t.pt")
    else:
        path = osp.join(ANALYSIS_DIR, "activations", f"{job}.pt")
    if osp.exists(path) and overwrite is False:
        return torch.load(path)
        
    gen_imgs, Y_gt = job_mgmt.get_synthetic_ds_for_job(job, overwrite=overwrite)
    dataset = job_mgmt.get_dataset_for_job(job)
    incv3 = load_inception_regressor(dataset=dataset, tuned=tuned, activations_only=True)
    outputs = job_mgmt.get_attributes_for_job(job)
    activations = []
    with torch.no_grad():
        for B in range(0,len(gen_imgs),bsz):
            activations.append(incv3(normalize_img(gen_imgs[B:B+bsz].cuda())).cpu())
    activations = torch.cat(activations, 0)
    torch.save(activations, path)
    return activations

def get_incv3_activations_of_ds(dataset, bsz=32, slurm=False, overwrite=False, tuned=True):
    if slurm is True:
        return analyze.submit_job(get_incv3_activations_of_ds, dataset, bsz=bsz, slurm=False, tuned=tuned,
            overwrite=overwrite, job_name=f"incv3_act_ds")

    if tuned:
        path = osp.join(ANALYSIS_DIR, "activations", f"{dataset}_t.pt")
    else:
        path = osp.join(ANALYSIS_DIR, "activations", f"{dataset}.pt")
    if osp.exists(path) and overwrite is False:
        return torch.load(path)
        
    dataloaders = dataloader.get_dataloaders_for_dataset(dataset, batch_size=bsz)
    incv3 = load_inception_regressor(dataset=dataset, tuned=tuned, activations_only=True)
    activations = []
    with torch.no_grad():
        for batch in dataloaders["val"]:
            activations.append(incv3(normalize_img(batch["image"].cuda())).cpu())
    activations = torch.cat(activations, 0)
    torch.save(activations, path)
    return activations

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Written by Dougal J. Sutherland.
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # Product is almost singular
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean)

def get_fid(job, tuned=True):
    dataset = job_mgmt.get_dataset_for_job(job)
    activations = get_incv3_activations_of_ds(dataset, tuned=tuned)
    m1 = activations.mean(0).numpy()
    s1 = np.cov(activations.numpy(), rowvar=False)
    activations = get_incv3_activations_of_G(job, tuned=tuned)
    m2 = activations.mean(0).numpy()
    s2 = np.cov(activations.numpy(), rowvar=False)
    fid = calculate_frechet_distance(m1,s1, m2,s2)
    return fid
