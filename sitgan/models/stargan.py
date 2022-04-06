import os
osp = os.path
import numpy as np
import torch
nn=torch.nn
F=nn.functional

import util, losses, optim
from monai.networks.nets import SEResNet50, DenseNet121
from models.common import OutputTransform, Encoder
from models.ipgan import Generator

def train_model(args, dataloaders):
    paths=args["paths"]
    loss_args=args["loss"]

    G,DR = build_starGAN(args)
    models = {'G':G,'DR':DR}
    optims = optim.get_starGAN_optims(models, args["optimizer"])

    max_epochs = args["optimizer"]["epochs"]
    global_step = 0

    cc_w = loss_args["reconstruction loss"]
    attr_w = loss_args["attribute loss"]
    gp_w = loss_args["gradient penalty"]
    R_w = loss_args["regressor loss"]

    G_fxn, D_fxn = losses.adv_loss_fxns(loss_args)
    intervals = {"train":1}
    
    G_adv_tracker = util.MetricTracker(name="G adversarial loss", function=G_fxn, intervals=intervals)
    D_adv_tracker = util.MetricTracker(name="D adversarial loss", function=D_fxn, intervals=intervals)
    cc_tracker = util.MetricTracker(name="cycle consistency", function=losses.L1_dist_mean, intervals=intervals)
    attr_tracker = util.MetricTracker(name="G attribute loss", function=losses.maskedMAE_sum, intervals=intervals)
    R_tracker = util.MetricTracker(name="regressor loss", function=losses.maskedMSE_sum, intervals=intervals)
    gp_tracker = util.MetricTracker(name="gradient penalty", function=losses.gradient_penalty, intervals={"train":1})
    total_G_tracker = util.MetricTracker(name="G loss", intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    metric_trackers = [G_adv_tracker, D_adv_tracker, cc_tracker, attr_tracker,
        total_G_tracker, R_tracker, gp_tracker, *list(SIT_trackers.values())]

    def process_minibatch_grad(batch, attr_targ, phase="train"):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt_filled = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        attr_targ = torch.where(torch.isnan(attr_targ), torch.rand_like(attr_targ)*2-1, attr_targ)
        dy = attr_targ - attr_gt_filled
        with torch.no_grad():
            _, true_r = DR(orig_imgs)
        fake_img, transforms = G(orig_imgs, dy, return_transforms=True)
        fake_d, fake_r = DR(fake_img)
        recon_img = G(fake_img, -dy)
        
        G_adv = G_adv_tracker(fake_d, phase=phase)
        cc = cc_tracker(recon_img, orig_imgs, phase=phase)
        attr = attr_tracker(fake_r - true_r, attr_targ - attr_gt, phase=phase)
        
        G_loss = G_adv + cc * cc_w + attr * attr_w
        G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=G_loss)
        backward(G_loss, optims["G"])
        total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)


        true_d, true_r = DR(orig_imgs)
        fake_d, _ = DR(fake_img.detach())
        D_adv = D_adv_tracker(fake_d, true_d, phase=phase)
        R = R_tracker(true_r, attr_gt, phase=phase)
        gp = gp_tracker(orig_imgs, fake_img, DR=DR, phase=phase)
        DR_loss = D_adv + R * R_w + gp * gp_w

        backward(DR_loss, optims["DR"])
        example_outputs = {"orig_imgs": orig_imgs, "fake_img": fake_img,
            "recon_img": recon_img, "transforms": transforms}

        return G_loss, DR_loss, example_outputs

    def process_minibatch_DR(batch, attr_targ, phase="train"):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt_filled = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        attr_targ = torch.where(torch.isnan(attr_targ), torch.rand_like(attr_targ)*2-1, attr_targ)
        dy = attr_targ - attr_gt_filled
        true_d, true_r = DR(orig_imgs)
        with torch.no_grad():
            fake_img = G(orig_imgs, dy)
        fake_d, _ = DR(fake_img)
        
        D_adv = D_adv_tracker(fake_d, true_d, phase=phase)
        R = R_tracker(true_r, attr_gt, phase=phase)
        gp = gp_tracker(orig_imgs, fake_img, DR=DR, phase=phase)
        DR_loss = D_adv + R * R_w + gp * gp_w

        backward(DR_loss, optims["DR"])
        return DR_loss


    def backward(loss, optim):
        optim.zero_grad()
        loss.backward()
        optim.step()

    for epoch in range(1,max_epochs+1):
        G.train()
        DR.train()
        for batch in dataloaders["train"]:
            global_step += 1
            attr_targ = next(dataloaders["train_attr"].__iter__()).cuda()
            if global_step % args["network"]["generator"]["optimizer step interval"] == 0:
                G_loss, DR_loss, example_outputs = process_minibatch_grad(
                    batch, attr_targ, phase="train")
            else:
                DR_loss = process_minibatch_DR(batch, attr_targ)
            if global_step % 200 == 0:
                break

        if example_outputs["transforms"] is None:
            transforms = None
        else:
            transforms = example_outputs["transforms"][:2]
        util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
            example_outputs["orig_imgs"][:2], example_outputs["fake_img"][:2],
            example_outputs["recon_img"][:2], transforms=transforms)

        if attr_tracker.is_at_min("train"):
            torch.save(G.state_dict(), osp.join(paths["weights dir"], "best_G.pth"))
            torch.save(DR.state_dict(), osp.join(paths["weights dir"], "best_DR.pth"))
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            tracker.update_at_epoch_end(phase="train")

        optim.update_SIT_weights(SIT_w, args["loss"])

    torch.save(G.state_dict(), osp.join(paths["weights dir"], "final_G.pth"))
    torch.save(DR.state_dict(), osp.join(paths["weights dir"], "final_DR.pth"))


def build_starGAN(args):
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
    DR = DiscrimRegressor(num_attributes=num_attributes,
        type=network_settings["discriminator"]["type"],
        pretrained=network_settings["discriminator"]["pretrained"])
    
    return G.cuda(), DR.cuda()


from models.common import ConvConv, ResConvs, Down, Up, UpCat
class ConditionalUNet(nn.Module):
    def __init__(self, num_attributes, outputs=None, num_res_units=4, min_channels=32,
            upsampling_type="bilinear"):
        super().__init__()
        out_channels = util.get_num_channels_for_outputs(outputs)

        N = num_attributes
        self.C = (min_channels, min_channels*2, min_channels*4, min_channels*8)
        self.conv1 = ConvConv(1, self.C[0])
        self.down1 = Down(self.C[0]+N, self.C[1])
        self.down2 = Down(self.C[1], self.C[2])
        self.down3 = Down(self.C[2], self.C[3])
        self.down4 = Down(self.C[3]+N, self.C[3])
        self.low_convs = ResConvs(num_res_units, self.C[3])
        self.up1 = UpCat(self.C[3]+N, self.C[2], cat_size=self.C[3], upsampling_type=upsampling_type)
        self.up2 = UpCat(self.C[2], self.C[1], cat_size=self.C[2], upsampling_type=upsampling_type)
        self.up3 = UpCat(self.C[1]+N, self.C[0], cat_size=self.C[1], upsampling_type=upsampling_type)
        self.up4 = Up(self.C[0]+N, self.C[0], upsampling_type=upsampling_type)
        self.to_tx = nn.Conv2d(self.C[0], out_channels, kernel_size=1)
        self.final_transforms = OutputTransform(outputs)

    def forward(self, x, y, return_transforms=False):
        y_ = y.view(-1,y.size(1),*[1,1]) * 10.
        x1 = self.conv1(x)
        x2 = self.down1(torch.cat([x1, y_.tile(*x1.shape[2:])], dim=1))
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(torch.cat([x4, y_.tile(*x4.shape[2:])], dim=1))
        z = self.low_convs(x5)
        z = self.up1(torch.cat([z, y_.tile(*z.shape[2:])], dim=1), x4)
        z = self.up2(z, x3)
        z = self.up3(torch.cat([z, y_.tile(*z.shape[2:])], dim=1), x2)
        z = self.up4(torch.cat([z, y_.tile(*z.shape[2:])], dim=1))
        transforms = self.to_tx(z)
        return self.final_transforms(x, transforms, return_transforms=return_transforms)

class DiscrimRegressor(nn.Module):
    def __init__(self, num_attributes, type, pretrained=False):
        super().__init__()
        if type == "DenseNet":
            self.net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_attributes+1, pretrained=pretrained)
        elif type == "SEResNet":
            self.net = SEResNet50(in_channels=1, num_classes=num_attributes+1, spatial_dims=2, pretrained=pretrained)
        elif type == "simple":
            self.net = Encoder(in_channels=1, out_dim=num_attributes+1)

    def forward(self, x):
        out = self.net(x)
        return out[:,0], out[:,1:]
