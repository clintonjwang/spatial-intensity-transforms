import argparse, os, yaml, shutil
osp = os.path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path')
    parser.add_argument('-o', '--out_dir')
    cmd_args = parser.parse_args()

    args = args_from_file(osp.expanduser(cmd_args.config_path), cmd_args)
    paths = args["paths"]
    if osp.exists(paths["job output dir"]):
        print("overwriting existing run")
        shutil.rmtree(paths["job output dir"])
    os.makedirs(paths["weights dir"], exist_ok=True)
    yaml.safe_dump(args, open(osp.join(paths["job output dir"], "config.yaml"), 'w'))
    return args

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
        args = {}

    if cmd_args is not None:
        for param in ["config_path", "out_dir"]:
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

def infer_missing_args(args):
    paths = args["paths"]
    paths["job output dir"] = osp.expanduser(args["out_dir"])
    paths["weights dir"] = osp.join(paths["job output dir"], "weights")
    for k in args["optimizer"]:
        if "learning rate" in k:
            args["optimizer"][k] = float(args["optimizer"][k])
    args["optimizer"]["weight decay"] = float(args["optimizer"]["weight decay"])
    if args["network"]["outputs"] is None:
        args["network"]["outputs"] = ""
