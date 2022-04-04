import os, pdb, gc
osp = os.path

import numpy as np
import torch
nn = torch.nn
F = nn.functional

from tqdm import tqdm
import args as args_module
import util, losses, optim
from models.common import SkipEncoder, CondSkipDecoder, OutputTransform, modify_model

def train_model(args, dataloaders):
    paths=args["paths"]

    G = build_CVAE(args)
    optimizer = optim.get_CVAE_optimizer(G, args["optimizer"])

    max_epochs = args["optimizer"]["epochs"]
    global_step = 0

    intervals = {"train":1}

    NLL_tracker = util.MetricTracker(name="NLL loss", function=losses.L2_square_dist, intervals=intervals)
    KL_tracker = util.MetricTracker(name="KL loss", function=losses.KLD_from_std_normal, intervals=intervals)
    loss_tracker = util.MetricTracker(name="total loss", intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    metric_trackers = [NLL_tracker, KL_tracker, loss_tracker, *list(SIT_trackers.values())]

    def process_minibatch(batch, attr_targ, phase):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        x1,x2,x3,mu, logvar = G.enc_forward(x=orig_imgs, y=attr_gt)
        z = (x1,x2,x3,G.sample_z(mu, logvar))
        KL = KL_tracker(mu, logvar, phase=phase)
        if np.random.rand() < 1.1:
            fake_imgs, transforms = G.dec_forward(z, y_t=attr_gt, x=orig_imgs, return_transforms=True)
            NLL = NLL_tracker(fake_imgs, orig_imgs, phase=phase)
            loss = NLL + KL
        # else:
        #     attr_targ = torch.where(torch.isnan(attr_targ), attr_gt, attr_targ)
        #     dy = attr_targ - attr_gt
        #     fake_imgs, transforms = G.dec_forward(z, y_t=attr_targ, x=orig_imgs, return_transforms=True)
        #     _,_,_,mu2, logvar2 = G.enc_forward(x=fake_imgs, y=attr_targ)
        #     KL2 = KL_tracker(mu2, logvar2, phase=phase)
        #     loss = KL + KL2 - ((mu2 - mu1).pow(2).mean((1,2,3)) * (attr_targ - attr_gt).pow(2)).mean()
        loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=loss)

        if np.isnan(loss.item()):
            raise ValueError
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tracker.update_with_minibatch(loss.item(), phase=phase)

        return {"orig_imgs": orig_imgs, "fake_imgs": fake_imgs, "transforms": transforms}

    for epoch in range(1,max_epochs+1):
        G.train()
        for batch in dataloaders["train"]:
            attr_targ = None#next(dataloaders["train_attr"].__iter__()).cuda()
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

        # G.eval()
        # with torch.no_grad():
        #     orig_imgs, attr_gt = batch["image"][:2].cuda(), batch["attributes"][:2].cuda()
        #     attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        #     dy = torch.randn_like(attr_gt)
        #     new_imgs, transforms = G(orig_imgs, y=attr_gt, dy=dy, return_transforms=True)
            
        # util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
        #     orig_imgs, new_imgs, transforms=transforms)

        if loss_tracker.is_at_min("train"):
            torch.save(G.state_dict(), osp.join(paths["weights dir"], "best_G.pth"))
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            tracker.update_at_epoch_end(phase="train")

        optim.update_SIT_weights(SIT_w, args["loss"])
    torch.save(G.state_dict(), osp.join(paths["weights dir"], "final_G.pth"))

def build_CVAE(args):
    network_settings = args["network"]
    num_attributes = len(args["data loading"]["attributes"])
    G = Generator(num_attributes=num_attributes, img_shape=args["data loading"]["image shape"],
        outputs=network_settings["outputs"], C_enc=network_settings["min channels"],
        C_dec=network_settings["min channels"])
    modify_model(network_settings["modifications"], [G])
    return G.cuda()

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
        assert img_shape[0] % 16 == 0 and img_shape[1] % 16 == 0
        self.z_dim = z_dim
        self.encoder = SkipEncoder(in_channels=1+num_attributes, out_dim=z_dim*2, C=C_enc)
        self.decoder = CondSkipDecoder(in_dim=z_dim, num_attributes=num_attributes,
            out_shape=img_shape, out_channels=out_channels, C=C_dec)
        self.final_transforms = OutputTransform(outputs)

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, y, dy=None, return_transforms=False):
        if dy is None:
            dy = torch.zeros_like(y)
        x1,x2,x3, mu, logvar = self.enc_forward(x, y)
        z = (x1,x2,x3,self.sample_z(mu, logvar))
        return self.dec_forward(z, y+dy, x, return_transforms=return_transforms)
    def enc_forward(self, x, y):
        xy = torch.cat([x, y.view(-1,y.size(1),1,1).tile(*x.shape[2:])], dim=1)
        x1,x2,x3,p_z_y = self.encoder(xy)
        return x1,x2,x3,p_z_y[:,:self.z_dim], p_z_y[:,self.z_dim:]
    def dec_forward(self, z, y_t, x=None, return_transforms=False):
        transforms = self.decoder(z, y_t)
        return self.final_transforms(x, transforms, return_transforms=return_transforms)
