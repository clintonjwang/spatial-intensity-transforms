import os, pdb, gc
osp = os.path

import numpy as np
import torch
nn = torch.nn
F = nn.functional
from monai.metrics import MSEMetric

from tqdm import tqdm
import args as args_module
import util, losses, optim
from models.common import Encoder, Decoder, OutputTransform

def train_model(args, dataloaders):
    paths=args["paths"]

    G = build_CVAE(args)
    optimizer = optim.get_CVAE_optimizer(G, args["optimizer"])

    max_epochs = args["optimizer"]["epochs"]
    global_step = 0

    intervals = {"train":1, "val":eval_interval}

    NLL_tracker = util.MetricTracker(name="NLL loss", function=MSEMetric(), intervals=intervals)
    KL_tracker = util.MetricTracker(name="KL loss", function=losses.KLD_from_std_normal, intervals=intervals)
    loss_tracker = util.MetricTracker(name="total loss", intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    metric_trackers = [NLL_tracker, KL_tracker, loss_tracker, *list(SIT_trackers.values())]

    def process_minibatch(batch, phase):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        mu, logvar = G.enc_forward(x=orig_imgs, y=attr_gt)
        z = G.sample_z(mu, logvar)
        recon_img, transforms = G.dec_forward(z, dy=torch.zeros_like(attr_gt), x=orig_imgs, return_transforms=True)
        NLL = NLL_tracker(recon_img, orig_imgs, phase=phase)
        KL = KL_tracker(mu, logvar, phase=phase)
        loss = NLL + KL
        loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=loss)
            
        if np.isnan(loss.item()):
            pdb.set_trace()
            raise ValueError(f"loss became NaN")

        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        example_outputs = {"orig_imgs": orig_imgs, "recon_img": recon_img, "transforms": transforms}
        loss_tracker.update_with_minibatch(loss.item(), phase=phase)
        return loss.item(), example_outputs

    for epoch in range(1,max_epochs+1):
        G.train()
        for batch in dataloaders["train"]:
            loss, example_outputs = process_minibatch(batch, phase="train")
            global_step += 1
            if global_step % 1000 == 0:
                break

        with torch.no_grad():
            orig_imgs, attr_gt = batch["image"][:2].cuda(), batch["attributes"][:2].cuda()
            attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
            dy = torch.randn_like(attr_gt)
            new_imgs, transforms = G(orig_imgs, y=attr_gt, dy=dy, return_transforms=True)
            
        util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
            example_outputs["orig_imgs"][:2], new_imgs, transforms=transforms)

        G.eval()
        for batch in dataloaders["val"]:
            with torch.no_grad():
                loss, example_outputs = process_minibatch(batch, phase="val")
            global_step += 1
            if global_step % 200 == 0:
                break

        # with torch.no_grad():
        #     orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        #     attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        #     dy = torch.randn_like(attr_gt)
        #     new_imgs, transforms = G(orig_imgs, y=attr_gt, dy=dy, return_transforms=True)
        # util.save_examples(epoch, paths["job output dir"]+"/imgs/val",
        #     example_outputs["orig_imgs"], new_imgs, transforms=transforms)

        if loss_tracker.is_at_min("val"):
            torch.save(G.state_dict(), osp.join(paths["weights dir"], f"best_G.pth"))
            # util.save_metric_histograms(metric_trackers, epoch=epoch, root=paths["job output dir"]+"/plots")
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            tracker.update_at_epoch_end(phase="val")
            tracker.update_at_epoch_end(phase="train")

        torch.save(G.state_dict(), osp.join(paths["weights dir"], f"{epoch}_G.pth"))

        optim.update_SIT_weights(SIT_w, args["loss"])

def build_CVAE(args):
    network_settings = args["network"]
    num_attributes = len(args["data loading"]["attributes"])
    return Generator(num_attributes=num_attributes, img_shape=args["data loading"]["image shape"],
        outputs=network_settings["outputs"], C_enc=network_settings["min channels"],
        C_dec=network_settings["min channels"]).cuda()

class Generator(nn.Module):
    def __init__(self, num_attributes, img_shape, outputs=None,
            C_enc=32, C_dec=32, z_dim=50):
        super().__init__()
        if "displacement" in outputs or "velocity" in outputs:
            if "," in outputs:
                out_channels = 3
            else:
                out_channels = 2
        else:
            out_channels = 1
        assert img_shape[0] % 8 == 0 and img_shape[1] % 8 == 0
        self.z_dim = z_dim
        self.encoder = Encoder(in_channels=1+num_attributes, out_dim=z_dim*2, C=C_enc)
        self.decoder = Decoder(in_dim=z_dim+num_attributes, out_shape=img_shape,
            out_channels=out_channels, C=C_dec)
        self.final_transforms = OutputTransform(outputs)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, y, dy=None, return_transforms=False):
        if dy is None:
            dy = torch.zeros_like(y)
        mu, logvar = self.enc_forward(x, y)
        z = self.sample_z(mu, logvar)
        return self.dec_forward(z, dy, x, return_transforms=return_transforms)
    def enc_forward(self, x, y):
        xy = torch.cat([x, y.view(-1,y.size(1),1,1).expand(-1,-1,*x.shape[2:])], dim=1)
        p_z_y = self.encoder(xy)
        return p_z_y[:,:self.z_dim], p_z_y[:,self.z_dim:]
    def dec_forward(self, z, dy, x=None, return_transforms=False):
        zy = torch.cat((z, dy), dim=1)
        transforms = self.decoder(zy)
        return self.final_transforms(x, transforms, return_transforms=return_transforms)
