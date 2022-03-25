import dill as pickle
import os, torch
osp = os.path
import numpy as np

ANALYSIS_DIR = osp.expanduser("~/code/sitgan/analysis")

def get_test_subjects_for_mrigenie():
    path = osp.join(ANALYSIS_DIR, "2d_mrigenie.dat")
    _, val_datalist = pickle.load(open(path, "rb"))
    test_ids = []
    for dp in val_datalist:
        dp_id = dp["ID"]
        if dp_id.endswith("_0"):
            test_ids.append(dp_id[:dp_id.rfind("_")])
    return test_ids

def get_midslice_for_subject(subject_id):
    path = osp.join(ANALYSIS_DIR, "2d_mrigenie.dat")
    _, val_datalist = pickle.load(open(path, "rb"))

    for dp in val_datalist:
        if dp["ID"].startswith(subject_id) and dp["ID"].endswith("_7"):
            return dp

def get_mrigenie_extrapolation_age(subject_id, G, transforms, model_type, path, overwrite=False):
    if osp.exists(path) and overwrite is False:
        outputs = pickle.load(open(path, "rb"))
    else:
        dp = get_midslice_for_subject(subject_id)
        dp = transforms(dp)
        age_mean, age_std = np.load(osp.join(ANALYSIS_DIR, "mrigenie_normalizations.npy"))[0]
        base_age = dp["attributes"][0].item() * age_std + age_mean
        if base_age < 50:
            gt_ix = 0
        elif base_age < 60:
            gt_ix = 1
        elif base_age < 70:
            gt_ix = 2
        elif base_age < 80:
            gt_ix = 3
        else:
            gt_ix = 4
        age_diffs = torch.cat((torch.arange(-40,-9,10), torch.arange(10,41,10)))[4-gt_ix:8-gt_ix]
        dY = age_diffs.cuda().unsqueeze(1).tile(3)
        dY[:,1:].zero_()
        X_0 = dp["image"].cuda().unsqueeze(0).tile(4,1,1,1)
        attr_gt = dp["attributes"].cuda()
        attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt).unsqueeze(0).tile(4,1)
        with torch.no_grad():
            if model_type == "CVAE":
                X_est = G(X_0, y=attr_gt, dy=dY/age_std)
            elif model_type == "CAAE":
                attr_targ = attr_gt + (dY/age_std)
                X_est = G(X_0, y=attr_targ)
            else:
                X_est = G(X_0, dY/age_std)
        outputs = {"subject_id":subject_id, "X_0": dp["image"].squeeze(),
            "X_est": X_est.cpu().squeeze(1), "dY": age_diffs, "gt_ix": gt_ix}

        pickle.dump(outputs, open(path, "wb"))

    return outputs

# paths:
#   # raw data dir: /data/vision/polina/projects/wmh/clintonw/raw_data
#   # spreadsheets:
#   #   root: /data/vision/polina/projects/wmh/clintonw/tables
#   #   MRI metadata table: mri_metadata.csv
#   #   clinical table: clin_vars.csv
#   #   slurm job history: runs.csv
#   #   GASROS phenotype csv: "Phenotypic Data_4_17_2013.csv"
#   #   GASROS phenotype csv2: 1400204_MITadditionalphenotypicdata.csv
#   #   MRI-GENIE phenotype csv: mrigenie_12sites_2018-10-05.csv
#   data root: /data/vision/polina/scratch/clintonw/datasets/MRI-GENIE
#   GASROS:
#     # dcm dir: /data/vision/polina/projects/wmh/clintonw/raw_data/2013_march_DWI
#     3D npy subdir: gasros_flair/correct_bias_field
#     2D npy subdir: 2d_gasros/npy
#     # 2D seg subdir: 2d_gasros/seg
#   MRI-GENIE:
#     # dcm dir:
#     #   path: /data/vision/polina/projects/wmh/clintonw/raw_data/dcm
#     #   attributes by path depth:
#     #   - clinical site
#     #   - SiGN ID
#     #   - sequence name
#     # nii dir:
#     #   path: /data/vision/polina/projects/wmh/clintonw/raw_data/nii
#     #   attributes by path depth:
#     #   - clinical site
#     #   - SiGN ID
#     #   - sequence name
#     3D npy subdir: mrigenie_flair/rescale_and_register
#     2D npy subdir: 2d_mrigenie/npy
# # variable ordering:
# #   CCS:
# #   - LAA
# #   - CEMajor
# #   - SAO
# #   - OTHER
# #   - UNDETERMINED
# #   EtOH:
# #   - Daily
# #   - Occasional
# #   - Never/Rare
# #   Race:
# #   - Caucasian
# #   - AfricanAmerican
# #   - Asian
# #   - Other
# #   Tobacco:
# #   - EVER
# #   - NEVER
# #   other:
# #   - 'Yes'
# #   - 'No'
# # clinical sites:
# # - '04'
# # - '01'
# # - '03'
# # - '06'
# # - '07'
# # - '13'
# # - '16'
# # - '18'
# # - '19'
# # - '20'
# # - '21'
# # - '23'