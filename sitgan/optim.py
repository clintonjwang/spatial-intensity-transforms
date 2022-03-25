import torch
import util, losses

def get_SIT_weights_and_trackers(loss_args, intervals):
    weights = {}
    trackers = {}
    if "smooth field" in loss_args:
        weights["F"] = loss_args["smooth field"]
        trackers["F"] = util.MetricTracker(name="smooth field", function=losses.total_variation_norm, intervals=intervals)
    if "sparse field" in loss_args:
        weights["F2"] = loss_args["sparse field"]
        trackers["F2"] = util.MetricTracker(name="sparse field", function=losses.L2_norm, intervals=intervals)
    if "sparse intensity" in loss_args:
        weights["dx"] = loss_args["sparse intensity"]
        trackers["dx"] = util.MetricTracker(name="sparse intensity", function=losses.L1_norm, intervals=intervals)

    return weights, trackers

def add_SIT_losses(SIT_w, SIT_trackers, transforms, output_type, G_loss, phase="train"):
    if "displacement" in output_type or "velocity" in output_type:
        if transforms.size(1) == 3:
            F = SIT_trackers["F"](transforms[:,1:], phase=phase)
            F2 = SIT_trackers["F2"](transforms[:,1:], phase=phase)
        else:
            F = SIT_trackers["F"](transforms, phase=phase)
            F2 = SIT_trackers["F2"](transforms, phase=phase)
        G_loss = G_loss + F * SIT_w["F"] + F2 * SIT_w["F2"]
    if "diffs" in output_type:
        dx = SIT_trackers["dx"](transforms[:,:1], phase=phase)
        G_loss = G_loss + dx * SIT_w["dx"]
    return G_loss

def update_SIT_weights(SIT_w, loss_args):
    for k in SIT_w:
        SIT_w[k] *= 1-loss_args["regularizer decay"]

def get_CAAE_optims(models, optimizer_settings):
    optim_class = torch.optim.Adam
    kwargs = {"weight_decay":optimizer_settings["weight decay"], "betas":(.5,.999)}
    G_optim = optim_class(models["G"].parameters(), lr=optimizer_settings["G learning rate"], **kwargs)
    Dz_optim = optim_class(models["Dz"].parameters(), lr=optimizer_settings["Dz learning rate"], **kwargs)
    Dimg_optim = optim_class(models["Dimg"].parameters(), lr=optimizer_settings["Dimg learning rate"], **kwargs)
    return {'G':G_optim, 'Dz':Dz_optim, 'Dimg':Dimg_optim}

def get_starGAN_optims(models, optimizer_settings):
    optim_class = torch.optim.Adam
    kwargs = {"weight_decay":optimizer_settings["weight decay"], "betas":(.5,.999)}
    G_optim = optim_class(models["G"].parameters(), lr=optimizer_settings["G learning rate"], **kwargs)
    DR_optim = optim_class(models["DR"].parameters(), lr=optimizer_settings["D/R learning rate"], **kwargs)
    return {'G':G_optim, 'DR':DR_optim}

def get_IPGAN_optims(models, optimizer_settings):
    optim_class = torch.optim.Adam
    kwargs = {"weight_decay":optimizer_settings["weight decay"], "betas":(.5,.999)}
    G_optim = optim_class(models["G"].parameters(), lr=optimizer_settings["G learning rate"], **kwargs)
    D_optim = optim_class(models["D"].parameters(), lr=optimizer_settings["D learning rate"], **kwargs)
    return {'G':G_optim, 'D':D_optim}

def get_CVAE_optimizer(G, optimizer_settings):
    optim_class = torch.optim.Adam
    kwargs = {"weight_decay":optimizer_settings["weight decay"], "betas":(.5,.999)}
    G_optim = optim_class(G.parameters(), lr=optimizer_settings["G learning rate"], **kwargs)
    return G_optim