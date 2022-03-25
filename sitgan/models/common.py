import torch
nn=torch.nn
F=nn.functional

from monai.networks.blocks import Warp, DVF2DDF

class OutputTransform(nn.Module):
    def __init__(self, outputs):
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

    def forward(self, x, transforms, return_transforms=False):
        if len(self.outputs) == 0:
            out = torch.sigmoid(transforms) * 1.1 - .01
            transforms = None
        elif len(self.outputs) == 1:
            if self.outputs[0] == "displacement":
                out = self.warp(x, transforms)
            elif self.outputs[0] == "velocity":
                ddf = self.DVF2DDF(transforms)
                out = self.warp(x, ddf)
            elif self.outputs[0] == "diffs":
                out = x + transforms
            else:
                raise NotImplementedError
        elif len(self.outputs) == 2:
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

class Encoder(nn.Module):
    def __init__(self, out_dim, C=32, in_channels=1):
        super().__init__()
        self.layers = nn.Sequential(Conv_BN_ReLU(in_channels,C),
            Down(C,C*2),
            Down(C*2,C*4),
            Down(C*4,C*8),
            nn.Flatten(), nn.LazyLinear(1024), nn.LeakyReLU(inplace=True),
            nn.Linear(1024,out_dim))
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_shape, out_channels=1, C=32):
        super().__init__()
        self.min_ch = C
        self.decode_latent = FC_BN_ReLU(in_dim, C*8*out_shape[0]*out_shape[1]//64)
        self.layers = nn.Sequential(
            Up(C*8,C*4),
            Up(C*4,C*2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv_BN_ReLU(C*2,C, k=1),
            nn.Conv2d(C,out_channels, kernel_size=3, padding=1),
        )
        self.out_shape = out_shape
    def forward(self, z):
        z = self.decode_latent(z)
        z = z.view(z.size(0), self.min_ch*8, self.out_shape[0]//8, self.out_shape[1]//8)
        return self.layers(z)


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
    def __init__(self, in_size, out_size=None, residual=False):
        super().__init__()
        if out_size is None or residual is True:
            out_size = in_size
        self.layers = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=False)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual is True:
            return x + self.layers(x)
        else:
            return self.layers(x)


class ResConvs(nn.Module):
    def __init__(self, num_res_units, size):
        super().__init__()
        layers = [ConvConv(size, residual=True)] * num_res_units
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
    def __init__(self, in_size, out_size, upsampling_type="bilinear"):
        super().__init__()
        if upsampling_type == "bilinear":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)#, padding=1)
        self.conv = ConvConv(in_size * 2, out_size)

    def forward(self, x1, x2):
        up = self.up(x1)
        return self.conv(torch.cat([up, x2], dim=1))
