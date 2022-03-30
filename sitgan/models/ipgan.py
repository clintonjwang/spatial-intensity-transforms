import os, pdb, gc
osp = os.path

import numpy as np
import torch
nn = torch.nn
F = nn.functional

from tqdm import tqdm
from monai.metrics import MAEMetric
import kornia.losses

import args as args_module
import util, losses, optim
from data.dataloader import get_dataloaders
from models.common import Conv_BN_ReLU, Down, SkipEncoder, CondSkipDecoder, OutputTransform

def train_model(args, dataloaders):
    paths=args["paths"]
    loss_args=args["loss"]

    G,D = build_IPGAN(args)
    models = {'G':G, 'D':D}
    optims = optim.get_IPGAN_optims(models, args["optimizer"])

    max_epochs = args["optimizer"]["epochs"]
    global_step = 0

    recon_w = loss_args["reconstruction loss"]
    id_w = loss_args["ID loss"]
    gp_w = loss_args["gradient penalty"]

    G_fxn, D_fxn = losses.adv_loss_fxns(loss_args)
    intervals = {"train":1, "val":1}
    
    G_adv_tracker = util.MetricTracker(name="G adversarial loss", function=G_fxn, intervals=intervals)
    D_adv_tracker = util.MetricTracker(name="D adversarial loss", function=D_fxn, intervals=intervals)
    D_diff_tracker = util.MetricTracker(name="D diff loss", function=lambda x: x.squeeze(), intervals=intervals)
    ID_tracker = util.MetricTracker(name="ID preservation loss", function=losses.ID_loss, intervals=intervals)
    recon_tracker = util.MetricTracker(name="reconstruction loss", function=losses.L1_dist, intervals=intervals)
    gp_tracker = util.MetricTracker(name="gradient penalty", function=losses.gradient_penalty_y, intervals={"train":1})
    total_G_tracker = util.MetricTracker(name="total G loss", intervals=intervals)
    SIT_w, SIT_trackers = optim.get_SIT_weights_and_trackers(args["loss"], intervals)
    metric_trackers = [G_adv_tracker, D_adv_tracker, recon_tracker, ID_tracker,
        gp_tracker, total_G_tracker, *list(SIT_trackers.values())]

    def process_minibatch(batch, phase):
        orig_imgs, attr_gt = batch["image"].cuda(), batch["attributes"].cuda()
        attr_gt = torch.where(torch.isnan(attr_gt), torch.randn_like(attr_gt), attr_gt)
        if np.random.rand() < .1:
            dy = torch.zeros_like(attr_gt)
            trans_img, transforms = G(orig_imgs, dy, return_transforms=True)
            recon = recon_tracker(trans_img, orig_imgs, phase=phase)
            G_loss = recon * recon_w
        else:
            dy = torch.randn_like(attr_gt) * args["data loading"]["spread"] - attr_gt
            trans_img, transforms = G(orig_imgs, dy, return_transforms=True)
            id_reg = ID_tracker(trans_img, orig_imgs, dy, phase=phase)
            G_loss = id_reg * id_w
        fake_d = D(trans_img, attr_gt + dy)
        G_loss = G_loss + G_adv_tracker(fake_d, phase=phase)
        G_loss = optim.add_SIT_losses(SIT_w, SIT_trackers, transforms=transforms,
                output_type=args["network"]["outputs"], G_loss=G_loss)

        backward(G_loss, optims["G"])

        true_d = D(orig_imgs, attr_gt)
        fake_d = D(trans_img.detach(), attr_gt + dy)
        D_adv = D_adv_tracker(fake_d, true_d, phase=phase)
        diff = D(orig_imgs, attr_gt + dy) - D(orig_imgs, attr_gt)
        D_diff = D_diff_tracker(diff)
        gp = gp_tracker(orig_imgs, trans_img, D=D, y=attr_gt, dy=dy, phase=phase)
        D_loss = D_adv + D_diff + gp * gp_w

        backward(D_loss, optims["D"])

        total_G_tracker.update_with_minibatch(G_loss.item(), phase=phase)
        example_outputs = {"orig_imgs": orig_imgs, "gen_imgs": trans_img, "transforms": transforms}
        return example_outputs

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
            example_outputs = process_minibatch(batch, phase="train")
            global_step += 1
            if global_step % 200 == 0:
                break

        if example_outputs["transforms"] is None:
            transforms = None
        else:
            transforms = example_outputs["transforms"][:1]
        util.save_examples(epoch, paths["job output dir"]+"/imgs/train",
            example_outputs["orig_imgs"][:1], example_outputs["gen_imgs"][:1], transforms=transforms)

        if total_G_tracker.is_at_min("train"):
            for m,model in models.items():
                torch.save(model.state_dict(), osp.join(paths["weights dir"], f"best_{m}.pth"))
            # util.save_metric_histograms(metric_trackers, epoch=epoch, root=paths["job output dir"]+"/plots")
        # if epoch % long_interval == 0:
        util.save_plots(metric_trackers, root=paths["job output dir"]+"/plots")

        for tracker in metric_trackers:
            tracker.update_at_epoch_end(phase="train")
            tracker.update_at_epoch_end(phase="val")

        optim.update_SIT_weights(SIT_w, args["loss"])

    for m,model in models.items():
        torch.save(model.state_dict(), osp.join(paths["weights dir"], f"final_{m}.pth"))

def build_IPGAN(args):
    network_settings = args["network"]
    num_attributes = len(args["data loading"]["attributes"])
    G = Generator(num_attributes=num_attributes, img_shape=args["data loading"]["image shape"],
        outputs=network_settings["outputs"], C_enc=network_settings["generator"]["min channels"],
        C_dec=network_settings["generator"]["min channels"])
    D = Discriminator(num_attributes=num_attributes, type=network_settings["discriminator"]["type"])

    return G.cuda(), D.cuda()

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
        self.img_shape = img_shape
        self.encoder = SkipEncoder(out_dim=z_dim, C=C_enc)
        self.decoder = CondSkipDecoder(in_dim=z_dim, num_attributes=num_attributes,
            out_shape=img_shape, out_channels=out_channels, C=C_dec)
        self.final_transforms = OutputTransform(outputs)

    def forward(self, x, y, return_transforms=False):
        z = self.enc_forward(x)
        return self.dec_forward(z,y,x, return_transforms=return_transforms)
    def enc_forward(self, x):
        return self.encoder(x)
    def dec_forward(self, z, y, x=None, return_transforms=False):
        transforms = self.decoder(z, y)
        return self.final_transforms(x, transforms, return_transforms=return_transforms)

from monai.networks.nets import DenseNet121, SEResNet50
class Discriminator(nn.Module):
    def __init__(self, num_attributes, type):
        super().__init__()
        self.first = nn.Sequential(Conv_BN_ReLU(1,16), nn.MaxPool2d(2))
        if type == "DenseNet":
            self.net = DenseNet121(spatial_dims=2, in_channels=16+num_attributes,
                out_channels=1, pretrained=False)
        elif type == "SEResNet":
            self.net = SEResNet50(in_channels=16+num_attributes, num_classes=1, spatial_dims=2, pretrained=False)
            # self.net = nn.Sequential(Conv_BN_ReLU(16+num_attributes,32), Down(32,64),
            #     Down(64,128), nn.Flatten(), nn.LazyLinear(1024), nn.LeakyReLU(inplace=True), nn.Linear(1024,1))
    def forward(self, x, y):
        x = self.first(x)
        z = torch.cat((x, y.view(y.size(0), -1, 1, 1).tile(1,1,x.size(2),x.size(3))), dim=1)
        return self.net(z)

