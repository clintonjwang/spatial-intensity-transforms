import os, submitit, functools, itertools, shutil
osp=os.path
import numpy as np
import pandas as pd
import torch
import kornia.losses
import dill as pickle
from glob import glob

import util
from data import dataloader, adni
from analysis import tables, inception, distributions
import jobs as job_mgmt
from data.transforms import get_transforms

ANALYSIS_DIR = osp.expanduser("~/code/sitgan/temp")

def submit_job(fxn, *args, job_name="unnamed", **kwargs):
    executor = submitit.AutoExecutor(folder=osp.join(ANALYSIS_DIR, "slurm", job_name))
    executor.update_parameters(
        name=job_name,
        slurm_partition="gpu",
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=16,
        slurm_exclude="bergamot,perilla,caraway,cassia,anise",
        slurm_exclusive=True,
        timeout_min=90000,
    )
    job = executor.submit(fxn, *args, **kwargs)
    return job

def submit_array_job(fxn, *iterated_args, job_name="unnamed", **kwargs):
    kw_fxn = functools.partial(fxn, **kwargs)
    executor = submitit.AutoExecutor(folder=osp.join(ANALYSIS_DIR, "slurm", job_name))
    executor.update_parameters(
        name=job_name,
        slurm_partition="gpu",
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=16,
        slurm_exclude="bergamot,perilla,caraway,cassia,anise",
        slurm_exclusive=True,
        timeout_min=90000,
    )
    jobs = executor.map_array(kw_fxn, *iterated_args)
    return jobs

def delete_jobs_without_results():
    results_folders = glob(osp.expanduser("~/code/sitgan/results/*"))
    for folder in results_folders:
        if not osp.exists(folder+"/weights/best_G.pth"):
            job = osp.basename(folder)
            print(job)
            shutil.rmtree(folder)

def collect_all_missing_job_metrics():
    results_folders = glob(osp.expanduser("~/code/sitgan/results/*"))
    df = tables.get_results_table()
    for folder in results_folders:
        job = osp.basename(folder)
        if job == "manual":
            continue
        elif (job not in df.index or np.isnan(df.loc[job, 'age mean error'])) and osp.exists(folder+"/weights/best_G.pth"):
            collect_metrics_for_jobs(job, slurm=True)
            print(job)


def collect_metrics_for_jobs(jobs, **kwargs):
    if isinstance(jobs, str):
        jobs = [jobs]
    get_regressor_metrics_for_job(jobs, **kwargs)
    get_image_fidelity_metrics_for_jobs(jobs, **kwargs)
collect_metrics_for_adni_jobs = collect_metrics_for_mrigenie_jobs = collect_metrics_for_jobs

def get_regressor_metrics_for_job(jobs, slurm=False, exact=True, overwrite=False):
    if isinstance(jobs, str):
        jobs = [jobs]
    if slurm is True:
        return submit_job(get_regressor_metrics_for_job, jobs,
            exact=exact, overwrite=overwrite, slurm=False, job_name="incv3_err")
    elif slurm == "array":
        return submit_array_job(get_regressor_metrics_for_job, jobs,
            exact=exact, overwrite=overwrite, slurm=False, job_name="incv3_err")

    if exact is False:
        jobs = util.flatten_list([util.get_repeated_jobs_from_base_job(job) for job in jobs])
    for job in jobs:
        outputs = job_mgmt.get_attributes_for_job(job)
        residuals = inception.get_inception_v3_residuals(job, overwrite=overwrite)
        df = tables.get_results_table()
        for var in outputs:
            df.loc[job, f"{var} mean error"] = np.nanmean(residuals[var])
            df.loc[job, f"{var} error STD"] = np.nanstd(residuals[var])
        tables.save_results_table(df)


def get_image_fidelity_metrics_for_jobs(jobs, slurm=False, exact=True, epoch=None, overwrite=True):
    if isinstance(jobs, str):
        jobs = [jobs]
    if slurm is True:
        return submit_job(get_image_fidelity_metrics_for_jobs,
            jobs, exact=exact, overwrite=overwrite, job_name="metrics")
    elif slurm == "array":
        return submit_array_job(get_image_fidelity_metrics_for_jobs,
            jobs, exact=exact, overwrite=overwrite, job_name="metrics")

    if exact is False:
        jobs = util.flatten_list([util.get_repeated_jobs_from_base_job(job) for job in jobs])

    for job in jobs:
        if "ADNI" in job_mgmt.get_dataset_for_job(job):
            path = osp.join(ANALYSIS_DIR, "rmse", f"{job}.bin")
            metrics = ("RMSE", "SSIM")
            if osp.exists(path) and overwrite is False:
                score_lists = pickle.load(open(path, "rb"))
            else:
                models, args = job_mgmt.get_job_model_and_args(job)
                _, transforms = get_transforms(args)
                score_lists = {metric:[] for metric in metrics}
                age_scale = np.load(osp.join(ANALYSIS_DIR, "adni_normalizations.npy"))[0,1]
                for subject_id in adni.get_test_subjects_for_adni():
                    scores = adni.get_image_matching_metrics_for_subject(subject_id=subject_id,
                        G=models["G"], transforms=transforms, model_type=args["network"]["type"], age_scale=age_scale,)
                    if scores is None:
                        continue
                    for ix,metric in enumerate(metrics):
                        score_lists[metric].append(scores[ix])
                pickle.dump(score_lists, open(path, "wb"))

            df = tables.get_results_table()
            for metric in metrics:
                df.loc[job, metric] = np.nanmean(score_lists[metric])
                df.loc[job, f"{metric} STD"] = np.nanstd(score_lists[metric])
            tables.save_results_table(df)

        else:
            path = osp.join(ANALYSIS_DIR, "fid", f"{job}.bin")
            if osp.exists(path) and overwrite is False:
                metrics = pickle.load(open(path, "rb"))
            else:
                # p_ut,r_ut = distributions.get_precision_and_recall(job, tuned=False, overwrite=overwrite)
                p,r = distributions.get_precision_and_recall(job, tuned=True, overwrite=overwrite)
                fid = inception.get_fid(job, tuned=True)
                #"FID":inception.get_fid(job, tuned=False),
                #"P_ut":p_ut, "R_ut":r_ut
                metrics = {"FID_tuned": fid, "F_{1/8}":p, "F_8":r,}
                pickle.dump(metrics, open(path, "wb"))

            df = tables.get_results_table()
            for metric in ("FID_tuned", "F_{1/8}", "F_8"):
                df.loc[job, metric] = metrics[metric]
            tables.save_results_table(df)
get_image_matching_metrics_for_jobs = get_image_fidelity_metrics_for_jobs

