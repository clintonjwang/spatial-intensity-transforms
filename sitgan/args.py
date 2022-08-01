import argparse, os, yaml, sys, shutil
osp = os.path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name')
    parser.add_argument('-s', '--slurm', action="store_true")
    parser.add_argument('-j', '--job_id', default="manual")
    parser.add_argument('-r', '--reps', default=1)
    # parser.add_argument("--ngpus", default=1, type=int)
    #parser.add_argument("-p", dest="partition", default="gpu", type=str) #gpu / QRTX5000 / A6000
    cmd_args = parser.parse_args()
    if cmd_args.slurm is True and cmd_args.job_id == "manual":
        cmd_args.job_id = cmd_args.config_name

    config_dir = osp.expanduser("~/code/sitgan/configs")
    config_name = cmd_args.job_id if cmd_args.config_name is None else cmd_args.config_name
    main_config_path = osp.join(config_dir, config_name+".yaml")

    args = args_from_file(main_config_path, cmd_args)
    paths = args["paths"]
    if osp.exists(paths["job output dir"]):
        shutil.rmtree(paths["job output dir"])
    os.makedirs(paths["weights dir"], exist_ok=True)
    yaml.safe_dump(args, open(osp.join(paths["job output dir"], "config.yaml"), 'w'))
    return args

def infer_missing_args(args):
    paths = args["paths"]
    paths["slurm output dir"] = osp.expanduser(paths["slurm output dir"])
    if "job_id" not in args:
        args["job_id"] = "manual"
        args["reps"] = 1
    if args["job_id"].startswith("lg_") or args["job_id"].startswith("A6"):
        args["partition"] = "A6000"
        args["data loading"]["batch size"] *= 4
    else:
        args["partition"] = "gpu"
    if args["reps"] > 1:
        args["random seed"] = list(range(1, args["reps"]+1))
        raise NotImplementedError("make different slurm dirs")
    paths["job output dir"] = osp.join(paths["slurm output dir"], args["job_id"])
    paths["loss history path"] = osp.join(paths["job output dir"], "metrics.csv")
    paths["weights dir"] = osp.join(paths["job output dir"], "weights")
    for k in args["optimizer"]:
        if "learning rate" in k:
            args["optimizer"][k] = float(args["optimizer"][k])
    args["optimizer"]["weight decay"] = float(args["optimizer"]["weight decay"])
    if args["network"]["outputs"] is None:
        args["network"]["outputs"] = ""

def merge_args(parent_args, child_args):
    if "_override_" in child_args.keys():
        return child_args
    for k,parent_v in parent_args.items():
        if k not in child_args.keys():
            child_args[k] = parent_v
        else:
            if isinstance(child_args[k], dict) and isinstance(parent_v, dict):
                child_args[k] = merge_args(parent_v, child_args[k])
    return child_args


def args_from_file(path, cmd_args=None):
    config_dir = osp.dirname(path)
    if osp.exists(path):
        args = yaml.safe_load(open(path, 'r'))
    else:
        jobname = osp.basename(path[:-5])
        if jobname.startswith("dms_"):
            c = jobname[jobname.find("_")+1:]
            args = {
                "parent": ["dit", "mrigenie", "stargan"],
                "loss": {
                    "sparse intensity": 10**float(c),
                }
            }
        elif jobname.startswith("das_"):
            c = jobname[jobname.find("_")+1:]
            args = {
                "parent": ["dit", "adni", "stargan"],
                "loss": {
                    "sparse intensity": 10**float(c),
                }
            }
        # elif jobname.endswith("_a_star"):
        #     c = jobname[:jobname.find("_")]
        #     args = {
        #         "parent": [c, "adni", "stargan"],
        #         "loss": {
        #             "attribute loss": 100,
        #             "reconstruction loss": .01,
        #         }
        #     }
        else:
            parents = []
            names = jobname.split("_")

            if names[0] in ["sit", "it", "st", "dt", "dit", "raw"]:
                parents.append(names[0])
            else:
                raise ValueError(f"bad path {path}")

            if names[1] == "m":
                parents.append("mrigenie")
            elif names[1] == "a":
                parents.append("adni")
            else:
                raise ValueError(f"bad path {path}")

            if names[2] == "star":
                parents.append("stargan")
            elif names[2] == "ipg":
                parents.append("ipgan")
            elif names[2] == "cvae":
                parents.append("cvae")
            elif names[2] == "caae":
                parents.append("caae")
            elif names[2] == "rgae":
                parents.append("rgae")
            else:
                raise ValueError(f"bad path {path}")

            args = {"parent":parents}

    if cmd_args is not None:
        for param in ["job_id", "config_name", "reps"]:#, "partition"]:
            args[param] = getattr(cmd_args, param)

    while "parent" in args:
        if isinstance(args["parent"], str):
            config_path = osp.join(config_dir, args.pop("parent")+".yaml")
            args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
        else:
            parents = args.pop("parent")
            for p in parents:
                config_path = osp.join(config_dir, p+".yaml")
                args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
            if "parent" in args:
                raise NotImplementedError("need to handle case of multiple parents each with other parents")

    config_path = osp.join(config_dir, "default.yaml")
    args = merge_args(yaml.safe_load(open(config_path, 'r')), args)
    infer_missing_args(args)
    return args
