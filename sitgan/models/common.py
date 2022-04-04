import torch
nn=torch.nn
F=nn.functional

from monai.networks.blocks import Warp, DVF2DDF
from composer import functional as cf

def modify_model(modifications, models):
    for m in models:
        if modifications["blurpool"]:
            cf.apply_blurpool(m)
        if modifications["squeeze-excite"]:
            cf.apply_squeeze_excite(m)
        if modifications["channels last"]:
            cf.apply_channels_last(m)
        if modifications["factorize layers"]:
            cf.apply_factorization(m)


class OutputTransform(nn.Module):
    def __init__(self, outputs, sigmoid=False):
        super().__init__()
        if outputs is None or outputs=="":
            self.outputs = []
        else:
            self.outputs = [s.strip() for s in outputs.split(",")]
        if "velocity" in outputs:
            self.DVF2DDF = DVF2DDF()
            self.warp = Warp()
        elif "displacement" in outputs:
            self.warp = Warp()
        self.sigmoid = sigmoid

    def forward(self, x, transforms, return_transforms=False):
        if len(self.outputs) == 0:
            if self.sigmoid:
                out = torch.sigmoid(transforms) * 1.5 - .25
            else:
                out = transforms
            if not self.training:
                out = torch.clamp(out, min=0, max=1)
            transforms = None
        elif len(self.outputs) == 1:
            if self.outputs[0] == "displacement":
                # transforms = transforms * 10
                out = self.warp(x, transforms)
            elif self.outputs[0] == "velocity":
                # transforms = transforms * 10
                ddf = self.DVF2DDF(transforms)
                out = self.warp(x, ddf)
            elif self.outputs[0] == "diffs":
                transforms = torch.tanh(transforms)
                out = x + transforms
            else:
                raise NotImplementedError
        elif len(self.outputs) == 2:
            # transforms = torch.cat((torch.tanh(transforms[:,:1]*.1), transforms[:,1:]*10), dim=1)
            transforms = torch.cat((torch.tanh(transforms[:,:1]), transforms[:,1:]), dim=1)
            dx, field = transforms[:,:1], transforms[:,1:]
            if self.outputs[0] == "diffs":
                x = x + dx
            else:
                raise NotImplementedError
                
            if self.outputs[1] == "displacement":
                out = self.warp(x, field)
            elif self.outputs[1] == "velocity":
                ddf = self.DVF2DDF(field)
                out = self.warp(x, ddf)
            else:
                raise NotImplementedError

        if return_transforms is True:
            return out, transforms
        else:
            return out

class SkipEncoder(nn.Module):
    def __init__(self, out_dim, C=32, in_channels=1):
        super().__init__()
        self.first = Conv_BN_ReLU(in_channels,C)
        self.d1 = Down(C,C*2)
        self.d2 = Down(C*2,C*4)
        self.d3 = Down(C*4,C*8)
        self.to_z = nn.Sequential(Down(C*8,C*8),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(), nn.Linear(C*8*16, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024,out_dim))
    def forward(self, x):
        x = self.first(x)
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        z = self.to_z(x3)
        return (x1, x2, x3, z)

class CondSkipDecoder(nn.Module):
    def __init__(self, in_dim, out_shape, num_attributes, out_channels=1, C=32, top_skip=True):
        super().__init__()
        self.min_ch = C
        self.out_shape = out_shape
        N = num_attributes
        self.decode_y = nn.Linear(N, in_dim, bias=False)
        self.decode_latent = FC_BN_ReLU(in_dim*2, C*8*out_shape[0]*out_shape[1]//256)
        self.up1 = UpCat(C*8, C*4, cat_size=C*8)
        self.up2 = UpCat(C*4+N, C*2, cat_size=C*4)
        if top_skip:
            self.up3 = UpCat(C*2+N, C, cat_size=C*2)
        else:
            self.up3 = Up(C*2+N, C)
        self.top_skip = top_skip
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, out_channels, kernel_size=1),
        )
    def forward(self, features, y):
        x1, x2, x3, z = features
        y_ = self.decode_y(y)
        z = self.decode_latent(torch.cat((z, y_), dim=1))
        z = z.view(z.size(0), self.min_ch*8, self.out_shape[0]//16, self.out_shape[1]//16)
        y = y.view(-1,y.size(1),1,1)
        o = self.up1(z,x3)
        o = self.up2(torch.cat([o, y.tile(*o.shape[2:])], dim=1), x2)
        if self.top_skip:
            o = self.up3(torch.cat([o, y.tile(*o.shape[2:])], dim=1), x1)
        else:
            o = self.up3(torch.cat([o, y.tile(*o.shape[2:])], dim=1))
        return self.final(o)

class FC_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_channels,out_channels),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)

class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_channels,out_channels, kernel_size=k, stride=s, padding=k//2),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.layers(x)


class ConvConv(nn.Module):
    def __init__(self, in_size, out_size=None):
        super().__init__()
        if out_size is None:
            out_size = in_size
        self.residual = (out_size == in_size)
        if out_size > in_size:
            self.layers = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, kernel_size=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)
            )
        else:
            self.layers = nn.Sequential(nn.Conv2d(in_size, in_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_size, out_size, kernel_size=1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.residual is True:
            return x + self.layers(x)
        else:
            return self.layers(x)


class ResConvs(nn.Module):
    def __init__(self, num_res_units, size):
        super().__init__()
        layers = [ConvConv(size)] * num_res_units
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)

class Down(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2), ConvConv(in_size, out_size)
        )
    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_size, out_size, upsampling_type="bilinear"):
        super().__init__()
        if upsampling_type == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)#, padding=1)
        self.conv = ConvConv(in_size, out_size)
    def forward(self, x):
        return self.conv(self.up(x))

class UpCat(nn.Module):
    def __init__(self, in_size, out_size, cat_size=None, upsampling_type="bilinear"):
        super().__init__()
        if cat_size is None:
            cat_size = in_size
        if upsampling_type == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)#, padding=1)
        self.conv = ConvConv(in_size + cat_size, out_size)

    def forward(self, x1, x2):
        up = self.up(x1)
        return self.conv(torch.cat([up, x2], dim=1))

class Encoder(nn.Module):
    def __init__(self, out_dim, C=32, in_channels=1):
        super().__init__()
        self.layers = nn.Sequential(
            Conv_BN_ReLU(in_channels,C),
            Down(C,C*2),
            Down(C*2,C*4),
            Down(C*4,C*8),
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(C*8*16, 1024), nn.LeakyReLU(inplace=True),
            nn.Linear(1024,out_dim))
    def forward(self, x):
        return self.layers(x)
