import pandas as pd
import numpy as np
import os, re

from am.dispositio.action import Action

def get_actions(A):
    return [
        (["build GASROS dataset", "rebuild GASROS dataset"], Action(build_gasros_dataset)),

        (["get all GASROS nii paths"], Action(get_all_gasros_nii_paths)),
        (["get segmentation path for GASROS datapoint"], Action(get_gasros_seg_path)),
        (["get nii path for GASROS datapoint"], Action(nii_convention_gasros)),
        (["get GASROS phenotype data"], Action(get_gasros_phenotype_table)),
        (["get GASROS WMHv table"], Action(get_gasros_wmhv_table)),

        # create datapoints
        (["set up 2D GASROS FLAIR pipeline"], Action(gasros_pipeline_2d)),
        (["set up 2D GASROS DWI pipeline"], Action(gasros_dwi_pipeline_2d)),
        (["get GASROS axial FLAIR datapoints"], Action(get_gasros_flair_datapoints)),
        (["set up 3D GASROS DWI pipeline"], Action(gasros_dwi_pipeline_3d)),
        # (["set up 3D GASROS FLAIR pipeline"], Action(gasros_pipeline_3d)),
        (["split 3D GASROS FLAIR dataset into 2D dataset"], Action(split_3D_gasros_dataset_into_2D_dataset)),
    ]

def set_seeds(A):
    A("remember path")("GASROS raw folder",
        "/data/vision/polina/projects/wmh/incoming/2018_05_15/gasros_files")

def correct_bad_2d_gasros_dps(A):
    dps = A("get datapoints with names")([f'axial_flair_affine_000000877_{i}' for i in range(17)])
    for dp in dps:
        img = A("load image")(dp["image path"])
        img["pixels"] /= 5
        A("save image")(img, dp["image path"])

# def gasros_pipeline_3d(A):
#     dataset = A("get active dataset")()
#     if dataset is None:
#         A("report failed precondition")("no active dataset")
#     A("new pipeline")("3D GASROS FLAIR pipeline")
#     A("set base image path for dataset")(load_function=A("getter")("nii path"), dataset=dataset)

#     pipeline = { # this makes everything except axial_flair_affine_000000877 worse...
#         "correct bias field": {
#             "actions": ["rescale images", "correct bias field"],
#         },
#     }
#     for name, kwargs in pipeline.items():
#         A("add pipeline step")(name=name, **kwargs)

def gasros_dwi_pipeline_3d(A, dataset=None):
    A("new pipeline")("3D GASROS DWI pipeline")
    A("set base image path for dataset")(load_function=A("getter")("image path"), dataset=dataset)

    pipeline = {
        "make isotropic": {
            "actions": ["move slice axis to back if needed",
                ("make isotropic", {"voxel length":1}),
            ],
            "convert to tensor": False,
        },

        "register": {
            "actions":[
                "rescale (0,1)",
                ("clamp",{"min":.05, "max":.98}),
                "rescale (0,1)",
                "ANTS registration to T1 atlas",
            ],
        },
    }
    for name, kwargs in pipeline.items():
        A("add pipeline step")(name=name, **kwargs)


def gasros_pipeline_2d(A, target_shape=(224,192)): #208,192
    dataset = A("get active dataset")()
    if dataset is None: A("failed precondition")("no active dataset")
    A("new pipeline")("2D MRI pipeline")

    A("set base image path for dataset")(load_function=A("getter")("image path"),
        dataset=dataset)

    pipeline = {
        "normalize npy": {
            "actions":[], #correct_bad_2d_gasros_dps
        },

        "crop npy": {
            "actions":[
                ("crop/pad to shape", {"shape":target_shape})],
        },
    }
    for name, kwargs in pipeline.items():
        kwargs = A("for each key")(kwargs, "replace spaces with _")
        A("add pipeline step")(name=name, **kwargs)

    A("mark unaugmented output pipeline step")("crop npy")
    A("mark output pipeline step")("normalize npy")

def gasros_dwi_pipeline_2d(A):
    return A("set up 2D GASROS FLAIR pipeline")(target_shape=(192,160))

def get_all_gasros_nii_paths(A, affine=True):
    paths = []
    if affine:
        for ix in range(1,1138):
            porpoise_id = A("format integer with zero-padding")(ix, 9)
            paths.append(A("join paths")("GASROS affine nii folder", "sub-{0}/sub-{0}_flair_atlas_ax_01_normalized.nii.gz".format(porpoise_id)))
    else:
        for ix in range(1,1138):
            porpoise_id = A("format integer with zero-padding")(ix, 9)
            paths.append(A("join paths")("GASROS nii folder", "sub-{0}/flair/sub-{0}_flair_ax_01.nii.gz".format(porpoise_id)))

    return paths

def get_gasros_seg_path(A, datapoint):
    return A("join paths")("GASROS affine nii folder", "sub-{0}/sub-{0}_flair_seg_manual_linear_ax_01.nii.gz".format(datapoint["Porpoise ID"]))

def build_gasros_dataset(A, dataset_name, **kwargs):
    if dataset_name == "GASROS FLAIR":
        paths = A("get all GASROS nii paths")(affine=True)
        dps = A("for each")(paths, "new MRI-GENIE datapoint from path")
        dataset = A("build dataset")(dps, name="GASROS FLAIR")
        dataset = add_clinical_vars_to_gasros(A, dataset, get_seg=False)
        A("exclude datapoints with condition")("is MRI-GENIE datapoint excluded")
        A("save dataset")(dataset_name)

    elif dataset_name == "2D GASROS":
        A("split 3D GASROS FLAIR dataset into 2D dataset")(**kwargs)

        #A("set up 2D GASROS pipeline")()
        #ds["processed image path"] = ds["image loader"] = None
        # bizarre bug requires...
        # step = A("get step")("augment")
        # if step["repetitions"]:
        #     step["image save path"] = lambda datapoint, repetition: A("join paths")(step["folder"], step["naming convention"](datapoint, repetition))
        # else:
        #     step["image save path"] = lambda datapoint: A("join paths")(step["folder"], step["naming convention"](datapoint))
        # ds = A("get active dataset")()
        # ds["image loader"] = ds["processed image path"] = step["image save path"]

    elif dataset_name == "GASROS DWI":
        img_paths = A("glob")(A["GASROS raw DWI folder"], "*", "*", "*", "*.img")
        datapoints = []
        for img_path in img_paths:
            subj_id, _, seq_name = img_path.split("/")[-4:-1]
            dp = A("instantiate")("datapoint", name=subj_id, **{
                "GASROS ID":A("get substring before delimiter")(subj_id, " "), "sequence name":seq_name,
                "image path": img_path,
                })
            datapoints.append(dp)

        dataset = A("new dataset from datapoints")(datapoints)
        dataset = add_clinical_vars_to_gasros(A, dataset, get_seg=False)
        A("save dataset")("GASROS DWI")

    elif dataset_name == "2D GASROS DWI":
        A("load dataset")("GASROS DWI")
        A("set up 3D GASROS DWI pipeline")()
        A("split 3D MRI dataset into 2D dataset")(A["GASROS 2D DWI npy folder"], input_step="register",
            n_slices=15, **kwargs)
        A("group K-fold")(n_splits=5, group_attribute="GASROS ID")
        A("assign nth cross-validation split")(1)
        A("save dataset")(dataset_name)

    else:
        A("TODO")()

def nii_convention_gasros(A, datapoint, affine=False):
    porpoise_id = datapoint["Porpoise ID"]
    if porpoise_id is None:
        return
    if affine:
        root = A("get known path")("GASROS affine nii folder")
    else:
        root = A["GASROS raw nii folder"]
    return A("join paths")(root, "sub-{0}/flair/sub-{0}_flair_ax_01.nii.gz".format(porpoise_id))

def get_gasros_flair_datapoints(A, affine=False):
    path = A("get path to dataset")("FLAIR GASROS")
    if not A("path exists")(path):
        A("create datapoints from all GASROS niis")(affine=affine)
    return A("load dataset")(path)

def add_clinical_vars_to_gasros(A, dataset, get_seg=False):
    conversion_path = A("join paths")("AIS spreadsheet folder", "160226_SiGN-ID-GASROS_conversion.xlsx")
    sign_gasros_table = A("load spreadsheet")(conversion_path)
    sign_to_gasros_dict = dict(sign_gasros_table.values)
    gasros_to_sign_dict = {g:s for s,g in sign_gasros_table.values}

    conversion_path = A("join paths")("AIS spreadsheet folder", "gasros_porpoise_conversion.csv")
    gasros_porpoise_table = A("load spreadsheet")(conversion_path)
    gasros_porpoise_table["GASROS_ID"] = gasros_porpoise_table["GASROS_ID"].apply(lambda s: s.replace("_", " "))
    gasros_to_porpoise_dict = dict(gasros_porpoise_table.values)
    porpoise_to_gasros_dict = {p:g for g,p in gasros_porpoise_table.values}

    porp_to_sign_dict = {str(p):str(gasros_to_sign_dict[g]) for p,g in porpoise_to_gasros_dict.items() if g in gasros_to_sign_dict}

    table = A("get all MRI-GENIE phenotype data with WMHv")()
    condition = lambda row: not isinstance(row["GASROS ID"], float)# and not A("is undefined")(row["Porpoise ID"])
    table = A("filter table by condition")(condition, inplace=True, table=table)
    fxn = lambda x: A("format integer with zero-padding")(x, 9) if not np.isnan(x) else x
    table = A("apply function to table column")("Porpoise ID", fxn)
    clin_vars = ["sex", "age", "NIHSS", "WMH volume (manual)", "WMH volume (auto)", "mRS"]

    for var in clin_vars:
        A("add variable to dataset")(var)

    for dp in A("get all datapoints")(dataset=dataset):
        if dp["Porpoise ID"] is not None:
            no_pad_porp = str(int(dp["Porpoise ID"]))
            if no_pad_porp in porp_to_sign_dict:
                dp["SiGN ID"] = porp_to_sign_dict[no_pad_porp]
        elif dp["GASROS ID"] is not None:
            if isinstance(dp["GASROS ID"], int):
                dp["GASROS ID"] = str(dp["GASROS ID"])
            if dp["GASROS ID"] in gasros_to_sign_dict:
                dp["SiGN ID"] = gasros_to_sign_dict[dp["GASROS ID"]]
        else:
            continue

        if get_seg:
            dp["segmentation path"] = A("get segmentation path for GASROS datapoint")(dp)

        valid_porpoises = table["Porpoise ID"].values
        valid_gasros = table["GASROS ID"].values
        if dp["Porpoise ID"] is not None and dp["Porpoise ID"] in valid_porpoises:
            row = A("get table row with condition")(lambda row: row["Porpoise ID"] == dp["Porpoise ID"])
        elif dp["GASROS ID"] is not None and dp["GASROS ID"] in valid_gasros:
            row = A("get table row with condition")(lambda row: row["GASROS ID"] == dp["GASROS ID"])
        # else:
        #     A("TODO")()
        if row is not None:
            for var in clin_vars:
                if var == "mRS":
                    dp["observations"][var] = row[var]
                else:
                    dp["observations"][var] = row[A("capitalize")(var)]

    if get_seg:
        dataset["segmentation loader"] = A("getter")("segmentation path")

    dataset["datapoints"] = A("select if")(dataset["datapoints"],
        lambda dp: not A("is None or NaN")(dp["observations"]["age"]))

    return dataset



#####################
### Clinical data
#####################

def get_gasros_phenotype_table(A):
    gasros_pheno_data1 = A("load table")(A("join paths")("AIS spreadsheet folder", 'Phenotypic Data_4_17_2013.csv'))
    gasros_pheno_data2 = A("load table")(A("join paths")("AIS spreadsheet folder", '1400204_MITadditionalphenotypicdata.csv'))

    A("clear merge rules")()
    match_condition = lambda f1, f2: str(f1)[:5] == str(f2)[:5]
    A("add match rule")(field1="Study ID", field2="Subject ID", new_name="GASROS ID", match_condition=match_condition)

    gasros_pheno_table = A("merge tables")(table1=gasros_pheno_data2, table2=gasros_pheno_data1)
    A("rename table column")("Study ID", "GASROS ID", table=gasros_pheno_table)

    col_map = {"Causative Clasification System (CCS) ": "CCS", "FU mRS":"mRS", "Hypertension":"HTN", "Diabetes Mellitus 2":"diabetes",
            "Atrial Fibrillation":"AFib",
            "Auto WMHv":"WMH volume (auto)", "Manual WMHv":"WMH volume (manual)"}

    A("rename table columns")(col_map, table=gasros_pheno_table)
    A("delete table columns")([r"Mix\Other Specify", "Ethnicity"], table=gasros_pheno_table)
    A("capitalize table column names")()
    return gasros_pheno_table

    # col_mapper = {"Sex":
    #     {1: "Male",
    #     2: "Female"},
    # "Race":
    #     {1: "White",
    #     2: "Black",
    #     3: "Asian",
    #     4: "Pacific Islander"
    #     5: "Native American"
    #     6: "Multiple Races"
    #     7: "Other",
    #     9999: "Unknown"},
    # "NIHSS":
    #     {9999:np.nan},
    # "mRS":
    #     {9999:np.nan},
    # "ETOH":
    #     {lambda x: x==1: "Never",
    #     lambda x: x>1: "Ever"},
    # "Tobacco":
    #     {lambda x: x==1: "Never",
    #     lambda x: x>1: "Ever"}
    # "CCS":
    #     {lambda x: x<=3: "LAA",
    #     lambda x: x>=4 and x<=6: "CAE",
    #     lambda x: x>=7 and x<=9: "SAO",
    #     lambda x: x>=10 and x<=12: "Other",
    #     lambda x: x>=13: "Undetermined",
    #     ".": None}}
    # A("load csv")(A["path to GASROS phenotype csv"])


def get_gasros_wmhv_table(A, refresh=False):
    fn = A("join paths")("AIS spreadsheet folder", 'gasros_wmhv.csv')

    if A("path exists")(fn) and not refresh:
        wmhv_table = A("load spreadsheet")(fn)
    else:

        wmhv_table = A("new table")(column_names=["Manual WMHv", "Auto WMHv"])
        for ix in range(1,1138):
            i = A("format integer with zero-padding")(ix, 9)

            path = A("join paths")("GASROS affine nii folder", "sub-{0}/sub-{0}_flair_seg_auto_linear_ax_01.nii.gz".format(i))
            image = A("load image")(path)
            wmhv_table.loc[ix,"Auto WMHv"] = image["pixels"].sum()/1000

            path = A("join paths")("GASROS affine nii folder", "sub-{0}/sub-{0}_flair_seg_manual_linear_ax_01.nii.gz".format(i))
            image = A("load image")(path)
            wmhv_table.loc[ix,"Manual WMHv"] = image["pixels"].sum()/1000

        A("save table")(fn=fn, table=wmhv_table)

    wmhv_table["Porpoise ID"] = wmhv_table.index

    return wmhv_table


def split_3D_gasros_dataset_into_2D_dataset(A, **kwargs):
    A("load dataset")("GASROS FLAIR")
    # A("set up 3D GASROS FLAIR pipeline")()
    A("split 3D MRI dataset into 2D dataset")(A("get known path")("GASROS 2D npy folder"), #input_step="correct bias field"
        load_path_fxn=A("getter")("nii path"), n_slices=17, **kwargs)
    A("group K-fold")(n_splits=5, group_attribute="Porpoise ID")
    A("assign nth cross-validation split")(1)
    A("save dataset")("2D GASROS")
    # A("add variable to dataset")("log(WMHa+1)")
    # area_fxn = lambda area: np.log(area+1) if not A("is None or NaN")(area) else area
    # for dp in A("get all datapoints")():
    #     dp["observations"]["log(WMHa+1)"] = area_fxn(dp["observations"]["WMH area"])
    #A("transform variable observations")("log(WMHa+1)", lambda x: x if A("is None or NaN")(x) else (x-mean)/stdev)
    #A("normalize dataset variable")("log(WMHa+1)")