import os, pdb, gc
osp = os.path

import numpy as np
import torch
nn=torch.nn
F=nn.functional

from tqdm import tqdm
from monai.metrics import MAEMetric, MSEMetric
import kornia.losses

import util, losses, optim
from monai.networks.nets import DenseNet, DenseNet121, UNet as MonaiUNet
from models.unet import UNet as MyUNet
from models.common import OutputTransform

def train_model(args, dataloaders):
    paths=args["paths"]
    loss_args=args["loss"]

    G,DR = build_starGAN(args)
    models = {'G':G,'DR':DR}
    optims = optim.get_starGAN_optims(models, args["optimizer"])

    max_epochs = args["optimizer"]["epochs"]
    gradscaler = torch.cuda.amp.GradScaler()
    global_step = 0

    cc_w = loss_args["reconstruction loss"]
    attr_w = loss_args["attribute loss"]
    gp_w = loss_args["gradient penalty"]
    R_w = loss_args["regressor loss"]

    G_fxn, D_fxn = losses.adv_loss_fxns(loss_args)
    intervals = {"train":1, "val":1}
    
    G_adv_tracker = util.MetricTracker(name="G adversarial loss", function=G_fxn, intervals=intervals)
    D_adv_tracker = util.MetricTracker(name="D adversarial loss", function=D_fxn, intervals=intervals)
    cc_tracker = util.MetricTracker(name="cycle consistency", function=MAEMetric(), intervals=intervals)
    attr_tracker = util.MetricTracker(name="G attribute loss", function=MSEMetric(), intervals=intervals)
    R_tracker = util.MetricTracker(name="regressor loss", function=losses.maskedMSE_sum, intervals=intervals)
    gp_tracker = util.MetricTracker(name="gradient penalty", function=losses.gradient_penalty, intervals={"train":1})
    total_G_tracker = util.MetricTracker(name="G loss", intervals=intervals)
    total_DR_tracker = util.MetricTracker(name="DR loss", intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    metric_trackers = [G_adv_tracker, D_adv_tracker, cc_tracker, attr_tracker,
        total_G_tracker, R_tracker, gp_tracker, total_DR_tracker, *list(SIT_trackers.values())]

    def process_minibatch_nograd(batch, phase="val"):
        with torch.no_grad():
            orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
            attr_gt_filled = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
            dy = torch.randn_like(attr_gt) * args["data loading"]["spread"] - attr_gt_filled
            true_d, true_r = DR(orig_imgs)
            fake_img, transforms = G(orig_imgs, dy, return_transforms=True)
            fake_d, fake_r = DR(fake_img)

            D_adv = D_adv_tracker(fake_d, true_d, phase=phase)
            R = R_tracker(true_r, attr_gt, phase=phase)
            DR_loss = D_adv + R * R_w

            recon_img = G(fake_img, -dy)
            
            G_adv = G_adv_tracker(fake_d, phase=phase)
            cc = cc_tracker(recon_img, orig_imgs, phase=phase)
            attr = attr_tracker(fake_r - true_r, dy, phase=phase)
            
            G_loss = G_adv + cc * cc_w + attr * attr_w
            G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                    output_type=args["network"]["outputs"], G_loss=G_loss)

            total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)
            total_DR_tracker.update_with_minibatch(DR_loss.item(), phase=phase)
            example_outputs = {"orig_imgs": orig_imgs, "fake_img": fake_img,
                "recon_img": recon_img, "transforms": transforms}

            return G_loss, DR_loss, example_outputs


    def process_minibatch_grad(batch, phase="train"):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt_filled = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        dy = torch.randn_like(attr_gt) * args["data loading"]["spread"] - attr_gt_filled
        with torch.no_grad():
            _, true_r = DR(orig_imgs)
        fake_img, transforms = G(orig_imgs, dy, return_transforms=True)
        fake_d, fake_r = DR(fake_img)

        recon_img = G(fake_img, -dy)
        
        G_adv = G_adv_tracker(fake_d, phase=phase)
        cc = cc_tracker(recon_img, orig_imgs, phase=phase)
        attr = attr_tracker(fake_r - true_r, dy, phase=phase)
        
        G_loss = G_adv + cc * cc_w + attr * attr_w

        G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=G_loss)

        backward(G_loss, optims["G"])

        true_d, true_r = DR(orig_imgs)
        fake_d, _ = DR(fake_img.detach())

        D_adv = D_adv_tracker(fake_d, true_d, phase=phase)
        R = R_tracker(true_r, attr_gt, phase=phase)
        DR_loss = D_adv + R * R_w
        gp = gp_tracker(orig_imgs, fake_img, DR=DR, phase=phase)
        DR_loss = DR_loss + gp * gp_w

        backward(DR_loss, optims["DR"])

        total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)
        total_DR_tracker.update_with_minibatch(DR_loss.item(), phase=phase)
        example_outputs = {"orig_imgs": orig_imgs, "fake_img": fake_img,
            "recon_img": recon_img, "transforms": transforms}

        if args["AMP"]:
            gradscaler.update()

        return G_loss, DR_loss, example_outputs

    def process_minibatch_DR(batch):
        phase="train"
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt_filled = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        dy = torch.randn_like(attr_gt) * args["data loading"]["spread"] - attr_gt_filled
        true_d, true_r = DR(orig_imgs)
        with torch.no_grad():
            fake_img = G(orig_imgs, dy)
        fake_d, _ = DR(fake_img)
        
        D_adv = D_adv_tracker(fake_d, true_d, phase=phase)
        R = R_tracker(true_r, attr_gt, phase=phase)
        gp = gp_tracker(orig_imgs, fake_img, DR=DR, phase=phase)
        DR_loss = D_adv + R * R_w + gp * gp_w

        total_DR_tracker.update_with_minibatch(DR_loss.item(), phase=phase)

        backward(DR_loss, optims["DR"])
        return DR_loss


    def backward(loss, optim):
        optim.zero_grad()
        if args["AMP"]:
            gradscaler.scale(loss).backward()
            gradscaler.step(optim)
        else:
            loss.backward()
            optim.step()


    for epoch in range(1,max_epochs+1):
        G.train()
        DR.train()
        epoch_iterator = tqdm(
            dataloaders["train"], desc="Training (Epoch X/X) (loss=X.X)", dynamic_ncols=True
        )
        for batch in epoch_iterator:
            global_step += 1
            if global_step % args["network"]["generator"]["optimizer step interval"] == 0:
                G_loss, DR_loss, example_outputs = process_minibatch_grad(
                    batch, phase="train")

                epoch_iterator.set_description(
                    "Training (Epoch %d/%d) (G_loss=%.3f, DR_loss=%.3f)" % (epoch, max_epochs, G_loss.item(), DR_loss.item())
                )

                if np.isnan(G_loss.item()) or np.isnan(DR_loss.item()):
                    pdb.set_trace()
                    raise ValueError(f"loss became NaN")
                    torch.save(G.state_dict(), osp.join(paths["weights dir"], "NaN_G.pth"))
                    torch.save(DR.state_dict(), osp.join(paths["weights dir"], "NaN_DR.pth"))

            else:
                if args["AMP"]:
                    with torch.cuda.amp.autocast():
                        DR_loss = process_minibatch_DR(batch)
                    gradscaler.update()
                else:
                    DR_loss = process_minibatch_DR(batch)

            if global_step % 1000 == 0:
                break

        if example_outputs["transforms"] is None:
            transforms = None
        else:
            transforms = example_outputs["transforms"][:1]
        util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
            example_outputs["orig_imgs"][:1], example_outputs["fake_img"][:1],
            example_outputs["recon_img"][:1], transforms=transforms)


        # gc.collect()
        # torch.cuda.empty_cache()
        epoch_iterator = tqdm(
            dataloaders["val"], desc="Validation (Epoch X/X) (G_loss=X.X, DR_loss=X.X)", dynamic_ncols=True
        )
        G.eval()
        DR.eval()
        for batch in epoch_iterator:
            G_loss, DR_loss, example_outputs = process_minibatch_nograd(batch, phase="val")
            epoch_iterator.set_description(
                "Validation (Epoch %d/%d) (G_loss=%.3f, DR_loss=%.3f)" % (epoch, max_epochs, G_loss.item(), DR_loss.item())
            )
            global_step += 1
            if global_step % 200 == 0:
                break
        util.save_examples(epoch, paths["job output dir"]+"/imgs/val",
            example_outputs["orig_imgs"], example_outputs["fake_img"],
            example_outputs["recon_img"], transforms=example_outputs["transforms"])

        if attr_tracker.is_at_min("val"):
            torch.save(G.state_dict(), osp.join(paths["weights dir"], "best_G.pth"))
            torch.save(DR.state_dict(), osp.join(paths["weights dir"], "best_DR.pth"))
            # util.save_metric_histograms(metric_trackers, epoch=epoch, root=paths["job output dir"]+"/plots")
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            tracker.update_at_epoch_end(phase="val")
            tracker.update_at_epoch_end(phase="train")

        attr_w += loss_args["attribute loss growth"]
        cc_w += loss_args["reconstruction loss growth"]
        optim.update_SIT_weights(SIT_w, args["loss"])
        
    torch.save(G.state_dict(), osp.join(paths["weights dir"], f"{epoch}_G.pth"))
    torch.save(DR.state_dict(), osp.join(paths["weights dir"], f"{epoch}_DR.pth"))



def build_starGAN(args):
    network_settings = args["network"]

    if network_settings["generator"]["type"] == "MONAI U-Net":
        UNetClass = MonaiUNet
    elif network_settings["generator"]["type"] == "My U-Net":
        UNetClass = MyUNet
    else:
        raise NotImplementedError("bad class "+network_settings["type"])

    if "3D" in args["dataset"]:
        spatial_dims = 3
    elif "2D" in args["dataset"]:
        spatial_dims = 2
    else:
        raise NotImplementedError

    num_attributes = len(args["data loading"]["attributes"])
    G = ConditionalUNet(num_attributes=num_attributes, UNetClass=UNetClass,
        outputs=network_settings["outputs"], spatial_dims=spatial_dims,
        min_channels=network_settings["generator"]["min channels"],
        num_res_units=network_settings["generator"]["res blocks"])
    DR = DiscrimRegressor(num_attributes=num_attributes, spatial_dims=spatial_dims,
        pretrained=network_settings["discriminator"]["pretrained"])

    return G.cuda(), DR.cuda()


class ConditionalUNet(nn.Module):
    def __init__(self, num_attributes, UNetClass, outputs=None, spatial_dims=2,
        num_res_units=4, min_channels=32):
        super().__init__()
        if "displacement" in outputs or "velocity" in outputs:
            if "," in outputs:
                out_channels = 3
            else:
                out_channels = 2
        else:
            out_channels = 1

        channels = (min_channels, min_channels*2, min_channels*4, min_channels*8)
        self.UNet = UNetClass(spatial_dims=spatial_dims, in_channels=num_attributes+1,
            out_channels=out_channels, channels=channels, strides=(2,2,2), #up_kernel_size=4,
            num_res_units=num_res_units)
        self.spatial_dims = spatial_dims
        self.final_transforms = OutputTransform(outputs)

    def forward(self, x, y, return_transforms=False):
        z = torch.cat([x, y.view(-1,y.size(1),*([1]*self.spatial_dims)).expand(-1,-1,*x.shape[2:])], dim=1)
        transforms = self.UNet(z)
        return self.final_transforms(x, transforms, return_transforms=return_transforms)


class DiscrimRegressor(nn.Module):
    def __init__(self, num_attributes, outputs=None, spatial_dims=2, pretrained=False):
        super().__init__()
        if spatial_dims == 2:
            self.net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_attributes+1, pretrained=pretrained)
        else:
            self.net = DenseNet(spatial_dims=3, in_channels=1, out_channels=num_attributes+1,
                init_features=48, growth_rate=16, block_config=(6, 12, 24))

    def forward(self, x):
        out = self.net(x)
        return out[:,0], out[:,1:]
