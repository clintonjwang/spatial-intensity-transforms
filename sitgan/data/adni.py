import pandas as pd
import numpy as np
import os, itertools
osp=os.path
import torch
import dill as pickle
import kornia.losses
from datetime import datetime

ANALYSIS_DIR = osp.expanduser("~/code/sitgan/analysis")

def get_test_subjects_for_adni():
    path = osp.join(ANALYSIS_DIR, "2d_adni.dat")
    _, val_datalist = pickle.load(open(path, "rb"))
    test_ids = set([dp["ID"][:10] for dp in val_datalist])
    return test_ids


def get_image_matching_metrics_for_subject(subject_id, G, transforms, n_attr,
        age_scale, model_type, max_samples=5):
    dps, T = get_all_timepoints_for_subject(subject_id)
    if len(dps) == 1:
        return
    dps = transforms(dps)
    combs = list(itertools.combinations(range(len(T)),2))
    timepoints = np.asarray(combs)[np.random.choice(len(combs), size=min(max_samples, len(combs)), replace=False)]

    dY, X_s, X_t, Y_s = [], [], [], []
    for i,j in timepoints:
        y = torch.stack([(T[j]-T[i]) / age_scale, *[torch.zeros_like(T[0])]*(n_attr-1)], 0)
        dY.append(y)
        X_s.append(dps[i]["image"])
        X_t.append(dps[j]["image"])
        Y_s.append(dps[i]["attributes"])
    X_s = torch.stack(X_s,0).cuda()
    X_t = torch.stack(X_t,0).cuda()
    dY = torch.stack(dY,0).cuda()
    with torch.no_grad():
        if model_type == "CVAE":
            Y_s = torch.stack(Y_s,0).cuda()
            X_est = G(X_s, y=Y_s, dy=dY)
        elif model_type == "CAAE":
            Y_t = torch.stack(Y_s,0).cuda()+dY
            X_est = G(X_s, y=Y_t)
        else:
            X_est = G(X_s, dY)

    def rmse_f(x1, x2):
        return (x1-x2).pow(2).view(x1.size(0), -1).mean(1).sqrt()
    rmse = rmse_f(X_est, X_t).mean(0)
    dssim = kornia.losses.ssim(X_est, X_t, window_size=11, reduction="mean")
    return rmse.item(), dssim.item()


def get_adni_trajectory(subject_id, G, transforms, n_attr, age_scale, model_type, path, overwrite=False):
    if osp.exists(path) and overwrite is False:
        outputs = pickle.load(open(path, "rb"))
    else:
        dps, T = get_all_timepoints_for_subject(subject_id)
        if len(dps) < 3:
            return
        # m,s = np.load(osp.join(ANALYSIS_DIR, "adni_normalizations.npy"))[0]
        # base_age = dps[0]["attributes"][0] * s + m
        cur_dt = 0
        ixs = list(range(len(T)))
        for i,dt in enumerate(T):
            if dt - cur_dt < .5:
                ixs.remove(i)
            else:
                cur_dt = dt

        if len(ixs) < 3:
            return
        if T[-1].item() < 2:
            return

        dps = transforms(dps)
        dY, X_gt = [], []
        X_0 = dps[0]["image"].cuda().unsqueeze(0).tile(len(ixs),1,1,1)
        for i in ixs:
            y = torch.stack([(T[i]-T[0])/age_scale, *[torch.zeros_like(T[0])]*(n_attr-1)], 0)
            dY.append(y)
            X_gt.append(dps[i]["image"])
        X_gt = torch.stack(X_gt,0)
        dY = torch.stack(dY,0).cuda()
        if model_type in ("CVAE", "CAAE"):
            attr_gt = dp["attributes"].cuda()
            attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt).unsqueeze(0).tile(4,1)
        with torch.no_grad():
            if model_type == "CVAE":
                X_est = G(X_0, y=attr_gt, dy=dY)
            elif model_type == "CAAE":
                attr_targ = attr_gt + dY
                X_est = G(X_0, y=attr_targ)
            else:
                X_est = G(X_0, dY)
        outputs = {"subject_id":subject_id, "X_0": dps[0]["image"].squeeze(),
            "X_gt": X_gt.squeeze(1), "X_est": X_est.cpu().squeeze(1),
            "dT": [t for ix,t in enumerate(T) if ix in ixs]}

        pickle.dump(outputs, open(path, "wb"))

    return outputs


def parse_date(date_str):
    return datetime.strptime(date_str, "%m/%d/%Y").date()

def get_all_timepoints_for_subject(subject_id):
    path = osp.join(ANALYSIS_DIR, "2d_adni.dat")
    _, val_datalist = pickle.load(open(path, "rb"))

    dps = []
    for dp in val_datalist:
        if dp["ID"].startswith(subject_id) and dp["ID"].endswith("_7"):
            dps.append(dp)
    dates = {dp["ID"]:parse_date(dp["date"]) for dp in dps}
    dps = sorted(dps, key=lambda dp: dates[dp["ID"]])
    t0 = dates[dps[0]["ID"]]
    # base_dx = torch.as_tensor(dps[0]["observations"]["baseline diagnosis"])
    dt = torch.as_tensor([(dates[dp["ID"]]-t0).days/365.25 for dp in dps])
    # x = torch.stack([torch.as_tensor(np.load(dp["image"])) for dp in dps], 0)

    # return image, dt, and base_dx as float tensors of size (T,H,W), (T-1,), (1,)
    return dps, dt.float()

