import os, pdb, gc
osp = os.path

import numpy as np
import torch
nn=torch.nn
F=nn.functional

import util, losses, optim
from monai.networks.nets import SEResNet50, DenseNet121
from models.common import OutputTransform, modify_model, Encoder
from models.ipgan import Generator
from models.stargan import ConditionalUNet

def train_model(args, dataloaders):
    paths=args["paths"]
    loss_args=args["loss"]

    G,R = build_RGAE(args)
    models = {'G':G,'R':R}
    optims = optim.get_RGAE_optims(models, args["optimizer"])

    max_epochs = args["optimizer"]["epochs"]
    global_step = 0

    cc_w = loss_args["reconstruction loss"]

    intervals = {"train":1}#, "val":1}
    attr_tracker = util.MetricTracker(name="G attribute loss", function=losses.maskedMAE_sum, intervals=intervals)
    cc_tracker = util.MetricTracker(name="cycle consistency", function=losses.L1_dist_mean, intervals=intervals)
    R_tracker = util.MetricTracker(name="regressor loss", function=losses.maskedMSE_sum, intervals=intervals)
    total_G_tracker = util.MetricTracker(name="G loss", intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    metric_trackers = [cc_tracker, attr_tracker, total_G_tracker, R_tracker, *list(SIT_trackers.values())]

    def process_minibatch(batch, attr_targ, phase="train"):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt_filled = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        attr_targ = torch.where(torch.isnan(attr_targ), torch.rand_like(attr_targ)*2-1, attr_targ)
        dy = attr_targ - attr_gt_filled
        with torch.no_grad():
            true_r = R(orig_imgs)
        fake_img, transforms = G(orig_imgs, dy, return_transforms=True)
        fake_r = R(fake_img)
        recon_img = G(fake_img, -dy)
        
        cc = cc_tracker(recon_img, orig_imgs, phase=phase)
        attr = attr_tracker(fake_r - true_r, attr_targ - attr_gt, phase=phase)
        G_loss = attr + cc * cc_w
        G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=G_loss)
        backward(G_loss, optims["G"])
        total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)

        true_r = R(orig_imgs)
        R_loss = R_tracker(true_r, attr_gt, phase=phase)

        if np.isnan(G_loss.item()) or np.isnan(R_loss.item()):
            raise ValueError("loss became NaN")
        backward(R_loss, optims["R"])
        return {"orig_imgs": orig_imgs, "fake_img": fake_img,
            "recon_img": recon_img, "transforms": transforms}

    def backward(loss, optim):
        optim.zero_grad()
        loss.backward()
        optim.step()

    for epoch in range(1,max_epochs+1):
        G.train()
        R.train()
        for batch in dataloaders["train"]:
            global_step += 1
            attr_targ = next(dataloaders["train_attr"].__iter__()).cuda()
            example_outputs = process_minibatch(
                batch, attr_targ, phase="train")
            if global_step % 200 == 0:
                break

        if example_outputs["transforms"] is None:
            transforms = None
        else:
            transforms = example_outputs["transforms"][:2]
        util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
            example_outputs["orig_imgs"][:2], example_outputs["fake_img"][:2], transforms=transforms)

        if attr_tracker.is_at_min("train"):
            torch.save(G.state_dict(), osp.join(paths["weights dir"], "best_G.pth"))
            torch.save(R.state_dict(), osp.join(paths["weights dir"], "best_R.pth"))
            # util.save_metric_histograms(metric_trackers, epoch=epoch, root=paths["job output dir"]+"/plots")
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            # tracker.update_at_epoch_end(phase="val")
            tracker.update_at_epoch_end(phase="train")

        optim.update_SIT_weights(SIT_w, args["loss"])

    torch.save(G.state_dict(), osp.join(paths["weights dir"], "final_G.pth"))
    torch.save(R.state_dict(), osp.join(paths["weights dir"], "final_R.pth"))


def build_RGAE(args):
    network_settings = args["network"]
    num_attributes = len(args["data loading"]["attributes"])
    if network_settings["generator"]["type"] == "AE":
        G = Generator(num_attributes=num_attributes, img_shape=args["data loading"]["image shape"],
            outputs=network_settings["outputs"], C_enc=network_settings["generator"]["min channels"],
            C_dec=network_settings["generator"]["min channels"])
    elif network_settings["generator"]["type"] == "UNet":
        G = ConditionalUNet(num_attributes=num_attributes,
            outputs=network_settings["outputs"],
            min_channels=network_settings["generator"]["min channels"],
            num_res_units=network_settings["generator"]["res blocks"])

    rtype=network_settings["regressor"]["type"]
    pretrained=network_settings["regressor"]["pretrained"]
    if rtype == "DenseNet":
        R = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_attributes, pretrained=pretrained)
    elif rtype == "SEResNet":
        R = SEResNet50(in_channels=1, num_classes=num_attributes, spatial_dims=2, pretrained=pretrained)
    elif rtype == "simple":
        R = Encoder(in_channels=1, out_dim=num_attributes)

    
    modify_model(network_settings["modifications"], (G, R))
    return G.cuda(), R.cuda()
