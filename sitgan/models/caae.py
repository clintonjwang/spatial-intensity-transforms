import os, pdb, gc
osp = os.path

import numpy as np
import torch
nn = torch.nn
F = nn.functional

from tqdm import tqdm

import args as args_module
import util, losses, optim
from models.common import FC_BN_ReLU, modify_model
from models.ipgan import Generator, Discriminator as img_Discrim


def train_model(args, dataloaders):
    paths=args["paths"]
    loss_args=args["loss"]

    G,Dz,Dimg = build_CAAE(args)
    models = {'G':G,'Dz':Dz,'Dimg':Dimg}
    optims = optim.get_CAAE_optims(models, args["optimizer"])

    max_epochs = args["optimizer"]["epochs"]
    gradscaler = torch.cuda.amp.GradScaler()
    global_step = 0

    recon_w = loss_args["reconstruction loss"]
    diff_w = loss_args["diff loss"]
    TV_w = loss_args["smooth output"]
    gp_w = loss_args["gradient penalty"]

    G_fxn, D_fxn = losses.adv_loss_fxns(loss_args)
    intervals = {"train":1}
    
    E_adv_tracker = util.MetricTracker(name="E adversarial loss", function=G_fxn, intervals=intervals)
    G_adv_tracker = util.MetricTracker(name="G adversarial loss", function=G_fxn, intervals=intervals)
    Dz_adv_tracker = util.MetricTracker(name="Dz adversarial loss", function=D_fxn, intervals=intervals)
    Dimg_adv_tracker = util.MetricTracker(name="Dimg adversarial loss", function=D_fxn, intervals=intervals)
    D_diff_tracker = util.MetricTracker(name="D diff loss", function=lambda x: x.squeeze(), intervals=intervals)
    recon_tracker = util.MetricTracker(name="reconstruction", function=losses.L1_dist_mean, intervals=intervals)
    TV_tracker = util.MetricTracker(name="smooth output", function=losses.total_variation_norm, intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    gp_tracker = util.MetricTracker(name="gradient penalty", function=losses.gradient_penalty_y, intervals={"train":1})
    total_G_tracker = util.MetricTracker(name="total G loss", intervals=intervals)
    metric_trackers = [E_adv_tracker, G_adv_tracker, Dz_adv_tracker, Dimg_adv_tracker,
        recon_tracker, TV_tracker, gp_tracker, total_G_tracker, *list(SIT_trackers.values())]

    def process_minibatch(batch, attr_targ, phase):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        x1,x2,x3,fake_z = G.enc_forward(orig_imgs)
        fake_dz = Dz(fake_z)
        E_adv = E_adv_tracker(fake_dz, phase=phase)

        if np.random.rand() < .2:
            dy = torch.zeros_like(attr_gt)
            fake_imgs, transforms = G.dec_forward((x1,x2,x3,fake_z), dy, x=orig_imgs, return_transforms=True)
            recon = recon_tracker(fake_imgs, orig_imgs, phase=phase)
            fake_dimg = Dimg(fake_imgs, attr_gt)
            G_loss = E_adv + recon * recon_w
        else:
            attr_targ = torch.where(torch.isnan(attr_targ), attr_gt, attr_targ)
            dy = attr_targ - attr_gt
            fake_imgs, transforms = G.dec_forward((x1,x2,x3,fake_z), dy, x=orig_imgs, return_transforms=True)
            fake_dimg = Dimg(fake_imgs, attr_targ) * (1+diff_w) - Dimg(fake_imgs, attr_gt) * diff_w
            G_adv = G_adv_tracker(fake_dimg, phase=phase)
            TV = TV_tracker(fake_imgs, phase=phase)
            G_loss = E_adv + G_adv + TV * TV_w
            G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                    output_type=args["network"]["outputs"], G_loss=G_loss)
            recon = None

        if np.isnan(G_loss.item()):
            raise ValueError("NaN loss")
        total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)
        backward(G_loss, optims["G"])

        uniform_z = torch.rand_like(fake_z)
        true_dz = Dz(uniform_z)
        fake_dz = Dz(fake_z.detach())
        Dz_adv = Dz_adv_tracker(fake_dz, true_dz, phase=phase)
        backward(Dz_adv, optims["Dz"])

        if recon is not None:
            return {"orig_imgs": orig_imgs, "fake_imgs": fake_imgs, "transforms": transforms}

        true_dimg = Dimg(orig_imgs, attr_gt)
        fake_dimg = Dimg(fake_imgs.detach(), attr_targ)
        Dimg_adv = Dimg_adv_tracker(fake_dimg, true_dimg, phase=phase)
        diff = Dimg(orig_imgs, attr_targ) - Dimg(orig_imgs, attr_gt)
        D_diff = D_diff_tracker(diff)
        Dimg_loss = Dimg_adv + D_diff * diff_w
        gp = gp_tracker(orig_imgs, fake_imgs, D=Dimg, y=attr_gt, dy=dy, phase=phase)
        Dimg_loss = Dimg_loss + gp * gp_w
        backward(Dimg_loss, optims["Dimg"])

        return {"orig_imgs": orig_imgs, "fake_imgs": fake_imgs, "transforms": transforms}

    def backward(loss, optim, retain_graph=False):
        optim.zero_grad()
        if args["AMP"]:
            gradscaler.scale(loss).backward(retain_graph=retain_graph)
            gradscaler.step(optim)
        else:
            loss.backward(retain_graph=retain_graph)
            optim.step()

    for epoch in range(1,max_epochs+1):
        for m in models.values():
            m.train()
        # attr_iter = dataloaders["train_attr"].__iter__()
        for batch in dataloaders["train"]:
            attr_targ = next(dataloaders["train_attr"].__iter__()).cuda()
            example_outputs = process_minibatch(batch, attr_targ, phase="train")
            global_step += 1
            if global_step % 200 == 0:
                break

        if example_outputs["transforms"] is None:
            transforms = None
        else:
            transforms = example_outputs["transforms"][:2]
        util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
            example_outputs["orig_imgs"][:2], example_outputs["fake_imgs"][:2], transforms=transforms)

        if total_G_tracker.is_at_min("train"):
            for m,model in models.items():
                torch.save(model.state_dict(), osp.join(paths["weights dir"], f"best_{m}.pth"))
            # util.save_metric_histograms(metric_trackers, epoch=epoch, root=paths["job output dir"]+"/plots")
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            tracker.update_at_epoch_end(phase="train")

        optim.update_SIT_weights(SIT_w, args["loss"])
        
    for m,model in models.items():
        torch.save(model.state_dict(), osp.join(paths["weights dir"], f"final_{m}.pth"))

def build_CAAE(args):
    network_settings = args["network"]
    num_attributes = len(args["data loading"]["attributes"])
    G = Generator(num_attributes=num_attributes, img_shape=args["data loading"]["image shape"],
        outputs=network_settings["outputs"], C_enc=network_settings["generator"]["min channels"],
        C_dec=network_settings["generator"]["min channels"])
    Dz = z_Discrim()
    Dimg = img_Discrim(num_attributes=num_attributes,
        type=network_settings["discriminator"]["type"],
        pretrained=network_settings["discriminator"]["pretrained"])

    modify_model(network_settings["modifications"], (G, Dz, Dimg))
    return G.cuda(), Dz.cuda(), Dimg.cuda()


class z_Discrim(nn.Module):
    def __init__(self, z_dim=50):
        super().__init__()
        self.layers = nn.Sequential(FC_BN_ReLU(z_dim,64), FC_BN_ReLU(64,32),
            FC_BN_ReLU(32,16), nn.Linear(16,1))
    def forward(self, x):
        return self.layers(x)
