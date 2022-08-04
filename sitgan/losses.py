import numpy as np 
import torch
nn=torch.nn 
F=nn.functional

<<<<<<< HEAD
import monai.losses
import monai.metrics

=======
>>>>>>> 05210ec13073bcca9b4dbff798fb626d963082dc
def total_variation_norm(img):
    B,C,H,W = img.shape
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return (tv_h+tv_w)/(B*C*H*W)

def L1_norm(img):
    return img.abs().sum((1,2,3))
def L1_norm_avg(img):
    return img.abs().mean((1,2,3))
def L2_norm(img):
    return img.pow(2).sum((1,2,3)).sqrt()
def L1_dist(img1, img2):
    return (img1-img2).abs().sum((1,2,3))
def L1_dist_mean(img1, img2):
    return (img1-img2).abs().mean((1,2,3))
def L2_dist(img1, img2):
    return (img1-img2).pow(2).sum((1,2,3)).sqrt()
def L2_square_dist(img1, img2):
    return (img1-img2).pow(2).sum((1,2,3))

def ID_loss(x_t, x_s, dy):
    age_diff = dy[:,0].abs()
    return (x_t - x_s).abs().mean((1,2,3)) * (-age_diff).exp()

def adv_loss_fxns(loss_settings):
    if "WGAN" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit.squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit - true_logit).squeeze()
        return G_fxn, D_fxn
    elif "standard" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit - torch.log1p(torch.exp(-fake_logit))#torch.log(1-torch.sigmoid(fake_logit)).squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit + torch.log1p(torch.exp(-fake_logit)) + torch.log1p(torch.exp(-true_logit))).squeeze()
<<<<<<< HEAD
        #-torch.log(1-fake_logit) - torch.log(true_logit)
=======
>>>>>>> 05210ec13073bcca9b4dbff798fb626d963082dc
        return G_fxn, D_fxn
    else:
        raise NotImplementedError

def gradient_penalty(real_img, generated_img, D=None, DR=None):
    B = real_img.size()[0]
    alpha = torch.rand(B, 1, 1, 1).expand_as(real_img).cuda()
    interp_img = nn.Parameter(alpha*real_img + (1-alpha)*generated_img.detach()).cuda()
    if DR is None:
        interp_logit = D(interp_img)
    else:
        interp_logit,_ = DR(interp_img)

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_img,
               grad_outputs=torch.ones(interp_logit.size()).cuda(),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2

def gradient_penalty_y(real_img, generated_img, D, y, dy):
<<<<<<< HEAD
=======
    #conditional discriminator version
>>>>>>> 05210ec13073bcca9b4dbff798fb626d963082dc
    B = real_img.size()[0]
    alpha = torch.rand(B, 1, 1, 1).expand_as(real_img).cuda()
    interp_img = nn.Parameter(alpha*real_img + (1-alpha)*generated_img.detach()).cuda()
    interp_logit = D(interp_img, y + (1-alpha[:,:,0,0])*dy)

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_img,
               grad_outputs=torch.ones(interp_logit.size()).cuda(),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2

def maskedMSE_sum(attr_pred, attr_gt):
    diffs = attr_pred - attr_gt
    return torch.where(torch.isnan(diffs), torch.zeros_like(diffs), diffs).pow(2).sum() / attr_pred.shape[0]

<<<<<<< HEAD
def maskedMAE_sum(attr_pred, attr_gt):#, slope=.5):
    diffs = attr_pred - attr_gt
    diffs = torch.where(torch.isnan(diffs), torch.zeros_like(diffs), diffs)
    #diffs = F.leaky_relu(diffs, slope=slope)
=======
def maskedMAE_sum(attr_pred, attr_gt):
    diffs = attr_pred - attr_gt
    diffs = torch.where(torch.isnan(diffs), torch.zeros_like(diffs), diffs)
>>>>>>> 05210ec13073bcca9b4dbff798fb626d963082dc
    return diffs.abs().sum() / attr_pred.shape[0]

def maskedMSE_mean(attr_pred, attr_gt):
    diffs = attr_pred - attr_gt
    diffs = diffs[~torch.isnan(diffs)]
    if torch.numel(diffs) == 0:
        return np.nan
    return diffs.pow(2).mean()

def KLD_from_std_normal(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

