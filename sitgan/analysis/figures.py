import os, torch, pdb
osp=os.path
F = torch.nn.functional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from data.transforms import get_transforms

from analysis import analyze, tables
from data import adni, mrigenie
import jobs as job_mgmt
import monai.transforms as mtr
ANALYSIS_DIR = osp.expanduser("~/code/sitgan/temp")

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_noclip = mtr.ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=255, clip=False, dtype=np.uint8)

def compile_trajectory_for_jobs_and_subject(jobs, subject_id):
    imgs = []
    for ix,job in enumerate(jobs):
        path = osp.join(ANALYSIS_DIR, "traj_figs", job, f"{subject_id}.png")
        if osp.exists(path):
            img = plt.imread(path)
            if "ADNI" in job_mgmt.get_dataset_for_job(job) and ix > 0:
                img = img[img.shape[0]//2:]
            imgs.append(img)
        else:
            print(f"could not find {path}")
    return np.concatenate(imgs, 0)

def save_adni_trajectories():
    jobs = [job for job in tables.get_all_adni_jobs_in_table() if not osp.exists(osp.join(ANALYSIS_DIR, "traj_figs", job))]
    analyze.submit_array_job(save_adni_trajectories_for_job, jobs, job_name="adni_trajs")
def save_mrigenie_age_extrapolations():
    jobs = [job for job in tables.get_all_mrigenie_jobs_in_table() if not osp.exists(osp.join(ANALYSIS_DIR, "traj_figs", job))]
    analyze.submit_array_job(save_mrigenie_age_extrapolations_for_job, jobs, job_name="mrig_trajs")

def save_sample_outputs_for_job(job, slurm=False, overwrite=False):
    if "ADNI" in job_mgmt.get_dataset_for_job(job):
        return save_adni_trajectories_for_job(job, slurm, overwrite)
    else:
        return save_mrigenie_age_extrapolations_for_job(job, slurm, overwrite)

def save_adni_trajectories_for_job(job, slurm=False, overwrite=False):
    if slurm is True:
        return analyze.submit_job(save_adni_trajectories_for_job, job,
            overwrite=overwrite, job_name="adni_traj")
    models, args = job_mgmt.get_job_model_and_args(job)
    _, transforms = get_transforms(args)
    age_scale = np.load(osp.join(ANALYSIS_DIR, "adni_normalizations.npy"))[0,1]
    os.makedirs(osp.join(ANALYSIS_DIR, "trajectories", job), exist_ok=True)
    os.makedirs(osp.join(ANALYSIS_DIR, "traj_figs", job), exist_ok=True)
    for subject_id in adni.get_test_subjects_for_adni():
        path = osp.join(ANALYSIS_DIR, "trajectories", job, f"{subject_id}.bin")
        outputs = adni.get_adni_trajectory(subject_id=subject_id, G=models["G"], transforms=transforms,
            path=path, age_scale=age_scale, model_type=args["network"]["type"], overwrite=overwrite)
        if outputs is None: continue
        X_0 = outputs["X_0"]
        top = torch.cat((X_0, *outputs["X_gt"]), dim=1)
        bot = torch.cat((X_0, *outputs["X_est"]), dim=1)
        full = torch.cat((top, bot), dim=0)
        full = rescale_clip(full)
        full[X_0.size(0):, :X_0.size(1)].zero_()
        img_path = path.replace("trajectories", "traj_figs").replace(".bin", ".png")
        plt.imsave(img_path, full, cmap="gray")
        
def save_mrigenie_age_extrapolations_for_job(job, slurm=False, overwrite=False):
    if slurm is True:
        return analyze.submit_job(save_mrigenie_age_extrapolations_for_job, job,
            overwrite=overwrite, job_name="mrig_traj")
    models, args = job_mgmt.get_job_model_and_args(job)
    _, transforms = get_transforms(args)
    age_scale = np.load(osp.join(ANALYSIS_DIR, "adni_normalizations.npy"))[0,1]
    os.makedirs(osp.join(ANALYSIS_DIR, "trajectories", job), exist_ok=True)
    os.makedirs(osp.join(ANALYSIS_DIR, "traj_figs", job), exist_ok=True)
    for subject_id in mrigenie.get_test_subjects_for_mrigenie():
        path = osp.join(ANALYSIS_DIR, "trajectories", job, f"{subject_id}.bin")
        outputs = mrigenie.get_mrigenie_extrapolation_age(subject_id, models["G"], transforms,
            path=path, model_type=args["network"]["type"], overwrite=overwrite)
        gt_ix = outputs["gt_ix"]
        img = torch.cat((*outputs["X_est"][:gt_ix], outputs["X_0"], *outputs["X_est"][gt_ix:]), dim=1)
        img = rescale_clip(img)
        img_path = path.replace("trajectories", "traj_figs").replace(".bin", ".png")
        plt.imsave(img_path, img, cmap="gray")
        

# def plot_empirical_dist(variable, phase=None, dataset=None):
#     observations = A("get observations of variable")(variable, phase=phase, dataset=dataset)
#     A("plot empirical distribution")(observations)
#     A("set axis label")(xlabel=A("get name")(variable), ylabel="Estimated PDF")

# def plot_predicted_distribution_of_variable_for_job(variable, job, dataset=None):
#     gt, predictions = A("get GT-prediction pairs of variable for job")(variable, job, dataset=dataset)
#     A("plot empirical distribution")(predictions)
#     A("set axis label")(xlabel=A("get name")(variable), ylabel="Estimated PDF")



from matplotlib.colors import LinearSegmentedColormap
def add_colorplot(pixels, abs_max=None):
    if abs_max is None:
        abs_max = pixels.abs().max().item()
    cdict = {'red':  ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.5, 0.9, 1.0),
                       (0.75, 1.0, 1.0),
                       (1.0, 0.4, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.5, 0.95, 0.95),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue':  ((0.0, 0.0, 0.4),
                       (0.25, 1.0, 1.0),
                       (0.5, 1.0, 0.9),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
            }
    blue_red_cmap = LinearSegmentedColormap('BlueRed1', cdict)
    fig,ax = plt.subplots()
    plt.imshow(pixels, cmap=blue_red_cmap, vmin=-abs_max, vmax=abs_max)
    plt.colorbar()
    ax.set_axis_off()
    fig.set_dpi(200)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    # array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # array = array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # array = torch.tensor(array).permute(-1,0,1).unsqueeze(0)
    # array = F.interpolate(array, scale_factor=pixels.size(0) / array.size(2))
    # plt.close()
    # return array.squeeze()

