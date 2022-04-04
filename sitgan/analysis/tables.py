import os
osp=os.path
import numpy as np
import pandas as pd
import torch

import util
import jobs as job_mgmt

ANALYSIS_DIR = osp.expanduser("~/code/sitgan/temp")
TABLE_PATH = osp.join(ANALYSIS_DIR, "results.csv")

def get_all_adni_jobs_in_table():
    df = get_results_table()
    return df[df["dataset"] == "2D ADNI T1"].index.values

def get_all_mrigenie_jobs_in_table():
    df = get_results_table()
    return df[df["dataset"] == "2D MRI-GENIE FLAIR"].index.values


def print_missing_model_types():
    fill_job_metadata()
    missing_jobs = {}
    df = get_results_table()

    model_types = ("CVAE", "CAAE", "IPGAN", "StarGAN")

    # ADNI baselines vs. SIT
    adni_raw_vs_sit = []
    for model_type in model_types:
        jobs = df[(df["model type"] == model_type) & (
            df["dataset"] == "2D ADNI T1") & (
            df["G outputs"] == "raw")].index
        if len(jobs) == 1:
            adni_raw_vs_sit.append(jobs.item())
        elif len(jobs) == 0:
            print(f"missing raw_a {model_type}")
        else:
            print(f"overlapping model types: {jobs.values}")

        jobs = df[(df["model type"] == model_type) & (
            df["dataset"] == "2D ADNI T1") & (
            df["G outputs"] == "diffs, velocity")].index
        if len(jobs) == 1:
            adni_raw_vs_sit.append(jobs.item())
        elif len(jobs) == 0:
            print(f"missing dit_a {model_type}")
        else:
            print(f"overlapping model types: {jobs.values}")

    # MRI-GENIE baselines vs. SIT
    mrig_raw_vs_sit = []
    for model_type in model_types:
        jobs = df[(df["model type"] == model_type) & (
            df["dataset"] == "2D MRI-GENIE FLAIR") & (
            df["G outputs"] == "raw")].index
        if len(jobs) == 1:
            mrig_raw_vs_sit.append(jobs.item())
        elif len(jobs) == 0:
            print(f"missing raw_m {model_type}")
        else:
            print(f"overlapping model types: {jobs.values}")

        jobs = df[(df["model type"] == model_type) & (
            df["dataset"] == "2D MRI-GENIE FLAIR") & (
            df["G outputs"] == "diffs, velocity")].index
        if len(jobs) == 1:
            mrig_raw_vs_sit.append(jobs.item())
        elif len(jobs) == 0:
            print(f"missing dit_m {model_type}")
        else:
            print(f"overlapping model types: {jobs.values}")


    # ADNI parameterizations
    G_out_types = ("diffs", "displacement", "diffs, displacement", "velocity")
    adni_G_outs = []
    for G_out_type in G_out_types:
        jobs = df[(df["model type"] == "StarGAN") & (
            df["dataset"] == "2D ADNI T1") & (
            df["G outputs"] == G_out_type)].index
        if len(jobs) == 1:
            adni_G_outs.append(jobs.item())
        elif len(jobs) == 0:
            print(f"missing a_star with G outputs: {G_out_type}")
        else:
            print(f"overlapping model types: {jobs.values}")


    # MRI-GENIE parameterizations
    mrig_G_outs = []
    for G_out_type in G_out_types:
        jobs = df[(df["model type"] == "StarGAN") & (
            df["dataset"] == "2D MRI-GENIE FLAIR") & (
            df["G outputs"] == G_out_type)].index
        if len(jobs) == 1:
            mrig_G_outs.append(jobs.item())
        elif len(jobs) == 0:
            print(f"missing m_star with G outputs: {G_out_type}")
        else:
            print(f"overlapping model types: {jobs.values}")

    for job in ["tuned_a_star", "tuned_m_star"]:
        if job not in df.index:
            print(f"missing {job}")


    # # ADNI fixed regularizer grid search
    # adni_fixed_reg_grid = []
    # # ADNI decaying regularizer grid search
    # adni_decay_reg_grid = []
    
    # # ADNI design choices/architectures (supplementary)
    # adni_techniques = []
    # # ADNI conditioning attributes (supplementary)
    # adni_attributes = []
    # # MRI-GENIE conditioning attributes (supplementary)
    # mrig_attributes = []


def fill_job_metadata():
    df = get_results_table()
    for job in df.index:
        for col in ("model type", "G outputs", "dataset", "smooth field", "regularizer decay"):
            if not isinstance(df.loc[job, col], str) and np.isnan(df.loc[job, col]):
                args = job_mgmt.get_job_args(job)
                df.loc[job, "model type"] = args["network"]["type"]
                if args["network"]["outputs"] == "" or not isinstance(args["network"]["outputs"], str):
                    df.loc[job, "G outputs"] = "raw"
                else:
                    df.loc[job, "G outputs"] = args["network"]["outputs"]
                df.loc[job, "dataset"] = args["dataset"]
                for term in ["smooth field", "sparse field", "sparse intensity", "regularizer decay"]:
                    if term in args["loss"]:
                        df.loc[job, term] = args["loss"][term]
                    else:
                        df.loc[job, term] = 0
                break
    save_results_table(df)

def any_column_is_missing_from_rows(columns, table):
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        if column not in table.columns or np.isnan(np.nanmean(table[column].values)):
            return True
    return False

def get_results_table():
    return pd.read_csv(TABLE_PATH, index_col="Job ID")
def save_results_table(table):
    table.to_csv(TABLE_PATH, index_label="Job ID")

def get_relevant_table_rows_for_job(table, job, exact=False):
    if exact is False:
        return table[table.index.str.startswith(job)]
    else:
        return table[table.index == job]


def get_metric_for_job(metric, job_name, average=True):
    table = get_results_table()
    relevant_rows = table[table.index.str.startswith(job_name)]
    if average:
        return np.nanmean(relevant_rows[metric].values)
    else:
        return [v for v in relevant_rows[metric].values if not np.isnan(v)]

def get_metric_values_for_job(metric, job_name):
    return get_metric_for_job(metric, job_name, average=False)

def report_latex_table_row_for_job(job, exact=True):
    dataset = jobs.get_dataset_for_job(job)
    if dataset == "2D ADNI T1":
        print(report_latex_adni_table_row_for_model(job, exact=exact))
    elif dataset == "2D MRI-GENIE FLAIR":
        print(report_latex_table_row_for_mrigenie_model(job, exact=exact))

def report_latex_adni_table_row_for_model(base_model, exact=True):
    relevant_rows = get_relevant_table_rows_for_job(table, base_model, exact=exact)
    if len(relevant_rows) == 0:
        print(f"no results for {base_model}")
        return

    columns = [str(relevant_rows["model type"].values[0])]

    if len(relevant_rows) == 1:
        if "ADNI RMSE" not in relevant_rows:
            print("missing RMSE")
            columns.append("N/A")
            columns.append("N/A")
        else:
            mean = relevant_rows["ADNI RMSE"].values[0]
            if "ADNI RMSE STD" in relevant_rows:
                std = relevant_rows["ADNI RMSE STD"].values[0]
                text = util.latex_mean_std(mean=mean, stdev=std, n_decimals=2)
            else:
                text = A("format float as string")(mean, n_decimals=3)
            columns.append(text)

            mean = relevant_rows["ADNI SSIM"].values[0]
            if "ADNI SSIM STD" in relevant_rows:
                std = relevant_rows["ADNI SSIM STD"].values[0]
                text = util.latex_mean_std(mean=mean, stdev=std, n_decimals=2)
            else:
                text = A("format float as string")(mean, n_decimals=3)
            columns.append(text)

        if "age mean error" not in relevant_rows:
            print("missing regressor error")
            columns.append("N/A")
        else:
            mean = relevant_rows["age mean error"].values[0]
            std = relevant_rows["age error STD"].values[0]
            text = util.latex_mean_std(mean=mean, stdev=std, n_decimals=2)
            columns.append(text)

    else:
        for metric in ("RMSE", "SSIM"):
            if metric in relevant_rows:
                text = util.latex_mean_std(relevant_rows[metric].values, n_decimals=2)
            else:
                text = "N/A"
            columns.append(text)

        if "age mean error" not in relevant_rows:
            print("missing regressor error")
            columns.append("N/A")
        else:
            age_mean_err = np.nanmean(relevant_rows["age mean error"].values)
            age_err_std = np.nanmean(relevant_rows["age error STD"].values)
            text = util.latex_mean_std(mean=age_mean_err, stdev=age_err_std, n_decimals=2)
            columns.append(text)

    return columns[0]+" & $" + "$ & $".join(columns[1:]) + r"$ \\"


def report_latex_table_row_for_mrigenie_model(job, exact=True):
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    relevant_rows = get_relevant_table_rows_for_job(table, job, exact=exact)
    if len(relevant_rows) == 0:
        print(f"no results for {job}")
        return

    columns = [str(relevant_rows["model type"].values[0])]
    for metric in ("FID",):
        if metric in relevant_rows:
            text = util.latex_mean_std(relevant_rows[metric].values, n_decimals=1)
        else:
            text = "N/A"
        columns.append(text)
    for metric in ("F_{1/8}", "F_8"):
        if metric in relevant_rows:
            text = util.latex_mean_std(relevant_rows[metric].values, n_decimals=2)
        else:
            text = "N/A"
        columns.append(text)

    if "age mean error" in relevant_rows:
        mean = np.nanmean(relevant_rows["age mean error"].values)
        std = np.nanmean(relevant_rows["age error STD"].values)
        text = util.latex_mean_std(mean=mean, stdev=std, n_decimals=2)
    else:
        text = "N/A"
    columns.append(text)

    return columns[0]+" & $" + "$ & $".join(columns[1:]) + r"$ \\"


def clear_metrics_for_job(job, exact=True):
    table = get_results_table()
    if exact:
        try:
            table.drop(job, inplace=True)
        except KeyError:
            print(f"missing entry {job}")
    else:
        for x in util.get_repeated_jobs_from_base_job(job):
            try:
                table.drop(x, inplace=True)
            except KeyError:
                print(f"missing entry {x}")
    save_results_table(table)


def tabulate_data_for_barplots(categories, mrigenie_jobs, adni_jobs):
    table = get_results_table()
    rows = []

    for ix, category in enumerate(categories):
        job = adni_jobs[ix]
        rows = table[table.index.str.startswith(job)]
        for value in np.nanmean(rows["RMSE"].values):
            rows.append([job, category, "2D ADNI T1", "RMSE", value])
        for value in np.nanmean(rows["SSIM"].values):
            rows.append([job, category, "2D ADNI T1", "DSSIM", value])
        for value in np.nanmean(rows["age mean error"].values):
            rows.append([job, category, "2D ADNI T1", "Age Error", value])
        for value in np.nanmean(rows["baseline diagnosis mean error"].values):
            rows.append([job, category, "2D ADNI T1", "Base-Dx Error", value])
        for value in np.nanmean(rows["MMSE mean error"].values):
            rows.append([job, category, "2D ADNI T1", "MMSE Error", value])
        for value in np.nanmean(rows["CDR mean error"].values):
            rows.append([job, category, "2D ADNI T1", "CDR Error", value])

        job = mrigenie_jobs[ix]
        rows = table[table.index.str.startswith(job)]
        for value in np.nanmean(rows["FID"].values):
            rows.append([job, category, "2D MRI-GENIE FLAIR", "FID", value])
        for value in np.nanmean(rows["F_{1/8}"].values):
            rows.append([job, category, "2D MRI-GENIE FLAIR", "Precision", value])
        for value in np.nanmean(rows["F_8"].values):
            rows.append([job, category, "2D MRI-GENIE FLAIR", "Recall", value])
        for value in np.nanmean(rows["age mean error"].values):
            rows.append([job, category, "2D MRI-GENIE FLAIR", "Age Error", value])
        for value in np.nanmean(rows["NIHSS mean error"].values):
            rows.append([job, category, "2D MRI-GENIE FLAIR", "NIHSS Error", value])
        for value in np.nanmean(rows["mRS mean error"].values):
            rows.append([job, category, "2D MRI-GENIE FLAIR", "mRS Error", value])

    table = pd.DataFrame(rows, columns=["Job ID", "Group", "Dataset", "Metric", "Value"])
    return table


# def recalculate_age_errors():
#     for ix, category in enumerate(categories):
#         job = adni_jobs[ix]
#         for job in util.get_repeated_jobs_from_base_job(job):
#             values = load_age_regressor_residuals(job, behavior_if_missing="compute")
#             if values is None:
#                 continue
#             for value in values:
#                 rows.append([job, category, "2D ADNI T1", "Age Error", value])

#         job = mrigenie_jobs[ix]
#         for job in util.get_repeated_jobs_from_base_job(job):
#             values = load_age_regressor_residuals(job, behavior_if_missing="compute")
#             if values is None:
#                 continue
#             for value in values:
#                 rows.append([job, category, "2D MRI-GENIE FLAIR", "Age Error", value])
