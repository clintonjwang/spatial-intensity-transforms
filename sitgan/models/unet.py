import torch, functools
nn=torch.nn
F=nn.functional

from models.common import ConvConv, ResConvs, Down, UpCat

class UNet(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=1, out_channels=1, channels=(64,128,256,512),
            strides=None, num_res_units=None, upsampling_type="bilinear"):
        super().__init__()
        self.C = channels
        self.conv1 = ConvConv(in_channels, self.C[0])
        self.down1 = Down(self.C[0], self.C[1])
        self.down2 = Down(self.C[1], self.C[2])
        self.down3 = Down(self.C[2], self.C[3])
        self.down4 = Down(self.C[3], self.C[3])
        self.low_convs = ResConvs(num_res_units, self.C[3])
        self.up1 = UpCat(self.C[3], self.C[2], upsampling_type=upsampling_type)
        self.up2 = UpCat(self.C[2], self.C[1], upsampling_type=upsampling_type)
        self.up3 = UpCat(self.C[1], self.C[0], upsampling_type=upsampling_type)
        self.up4 = UpCat(self.C[0], self.C[0], upsampling_type=upsampling_type)
        self.out = nn.Conv2d(self.C[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
