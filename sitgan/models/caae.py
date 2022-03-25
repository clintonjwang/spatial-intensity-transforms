import os, pdb, gc
osp = os.path

import numpy as np
import torch
nn = torch.nn
F = nn.functional

from tqdm import tqdm
from monai.metrics import MSEMetric

import args as args_module
import util, losses, optim
from models.common import FC_BN_ReLU
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
    TV_w = loss_args["smooth output"]
    # gp_w = loss_args["gradient penalty"]

    G_fxn, D_fxn = losses.adv_loss_fxns(loss_args)
    intervals = {"train":1, "val":1}
    
    E_adv_tracker = util.MetricTracker(name="E adversarial loss", function=G_fxn, intervals=intervals)
    G_adv_tracker = util.MetricTracker(name="G adversarial loss", function=G_fxn, intervals=intervals)
    Dz_adv_tracker = util.MetricTracker(name="Dz adversarial loss", function=D_fxn, intervals=intervals)
    Dimg_adv_tracker = util.MetricTracker(name="Dimg adversarial loss", function=D_fxn, intervals=intervals)
    recon_tracker = util.MetricTracker(name="reconstruction", function=MSEMetric(), intervals=intervals)
    TV_tracker = util.MetricTracker(name="smooth output", function=losses.total_variation_norm, intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    gp_tracker = util.MetricTracker(name="gradient penalty", function=losses.gradient_penalty, intervals={"train":1})
    total_G_tracker = util.MetricTracker(name="total G loss", intervals=intervals)
    metric_trackers = [E_adv_tracker, G_adv_tracker, Dz_adv_tracker, Dimg_adv_tracker,
        recon_tracker, TV_tracker, gp_tracker, total_G_tracker, *list(SIT_trackers.values())]

    def process_minibatch(batch, phase):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        fake_z = G.enc_forward(orig_imgs)
        recon_img, transforms = G.dec_forward(fake_z, attr_gt, x=orig_imgs, return_transforms=True)
        recon = recon_tracker(recon_img, orig_imgs, phase=phase)

        fake_dz = Dz(fake_z)
        E_adv = E_adv_tracker(fake_dz, phase=phase)
        fake_dimg = Dimg(recon_img, attr_gt)
        G_adv = G_adv_tracker(fake_dimg, phase=phase)
        G_loss = E_adv + G_adv + recon * recon_w
        if TV_w > 0:
            TV = TV_tracker(recon_img, phase=phase)
            G_loss = G_loss + TV * TV_w

        uniform_z = torch.rand_like(fake_z)*2-1
        true_dz = Dz(uniform_z)
        Dz_adv = Dz_adv_tracker(fake_dz, true_dz, phase=phase)

        true_dimg = Dimg(orig_imgs, attr_gt)
        Dimg_adv = Dimg_adv_tracker(fake_dimg, true_dimg, phase=phase)

        G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=G_loss)
            
        L = {"G":G_loss, "Dz":Dz_adv, "Dimg":Dimg_adv}
        total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)
        example_outputs = {"orig_imgs": orig_imgs, "recon_img": recon_img, "transforms": transforms}
        return L, example_outputs

    def process_minibatch_grad(batch, phase):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        fake_z = G.enc_forward(orig_imgs)
        recon_img, transforms = G.dec_forward(fake_z, attr_gt, x=orig_imgs, return_transforms=True)
        recon = recon_tracker(recon_img, orig_imgs, phase=phase)

        fake_dz = Dz(fake_z)
        E_adv = E_adv_tracker(fake_dz, phase=phase)
        fake_dimg = Dimg(recon_img, attr_gt)
        G_adv = G_adv_tracker(fake_dimg, phase=phase)
        G_loss = E_adv + G_adv + recon * recon_w
        if TV_w > 0:
            TV = TV_tracker(recon_img, phase=phase)
            G_loss = G_loss + TV * TV_w

        uniform_z = torch.rand_like(fake_z)*2-1
        G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=G_loss)
            
        backward(G_loss, optims["G"])

        true_dz = Dz(uniform_z)
        Dz_adv = Dz_adv_tracker(fake_dz, true_dz, phase=phase)

        true_dimg = Dimg(orig_imgs, attr_gt)
        Dimg_adv = Dimg_adv_tracker(fake_dimg, true_dimg, phase=phase)

        L = {"G":G_loss, "Dz":Dz_adv, "Dimg":Dimg_adv}
        # backward(L["Dz"], optims["Dz"])
        # backward(L["Dimg"], optims["Dimg"])
        total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)
        example_outputs = {"orig_imgs": orig_imgs, "recon_img": recon_img, "transforms": transforms}
        #gp_tracker?
        # backward(L["G"], optims["G"], retain_graph=True)

        return L, example_outputs

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
        for batch in dataloaders["train"]:
            L, example_outputs = process_minibatch_grad(batch, phase="train")
            # if args["debug"] is True:
            global_step += 1
            if global_step % 1000 == 0:
                break

        with torch.no_grad():
            orig_imgs, attr_gt = batch["image"][:2].cuda(), batch["attributes"][:2].cuda()
            attr_new = torch.randn_like(attr_gt)
            new_img, transforms = G(orig_imgs, attr_new, return_transforms=True)
            
        util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
            example_outputs["orig_imgs"][:2], new_img, transforms=transforms)

        # if epoch % eval_interval == 0 or epoch == max_epochs:
        #     gc.collect()
        #     torch.cuda.empty_cache()
        for m in models.values():
            m.eval()
        for batch in dataloaders["val"]:
            with torch.no_grad():
                L, example_outputs = process_minibatch(batch, phase="val")
            # if args["debug"] is True:
            global_step += 1
            if global_step % 200 == 0:
                break

        # with torch.no_grad():
        #     orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        #     attr_new = torch.randn_like(attr_gt)
        #     new_img, transforms = G(orig_imgs, attr_new, return_transforms=True)
        # util.save_examples(epoch, paths["job output dir"]+"/imgs/val",
        #     example_outputs["orig_imgs"], new_img, transforms=transforms)

        if total_G_tracker.is_at_min("val"):
            for m,model in models.items():
                torch.save(model.state_dict(), osp.join(paths["weights dir"], f"best_{m}.pth"))
            # util.save_metric_histograms(metric_trackers, epoch=epoch, root=paths["job output dir"]+"/plots")
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            tracker.update_at_epoch_end(phase="train")
            tracker.update_at_epoch_end(phase="val")

        for m,model in models.items():
            torch.save(model.state_dict(), osp.join(paths["weights dir"], f"{epoch}_{m}.pth"))

        optim.update_SIT_weights(SIT_w, args["loss"])
        
def build_CAAE(args):
    network_settings = args["network"]
    num_attributes = len(args["data loading"]["attributes"])
    G = Generator(num_attributes=num_attributes, img_shape=args["data loading"]["image shape"],
        outputs=network_settings["outputs"], C_enc=network_settings["generator"]["min channels"],
        C_dec=network_settings["generator"]["min channels"])
    D_z = z_Discrim()
    D_img = img_Discrim(num_attributes=num_attributes)

    return G.cuda(), D_z.cuda(), D_img.cuda()


class z_Discrim(nn.Module):
    def __init__(self, z_dim=50):
        super().__init__()
        self.layers = nn.Sequential(FC_BN_ReLU(z_dim,64), FC_BN_ReLU(64,32),
            FC_BN_ReLU(32,16), nn.Linear(16,1))
    def forward(self, x):
        return self.layers(x)
