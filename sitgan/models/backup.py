import torch
nn=torch.nn
F=nn.functional

class SkipDecoder(nn.Module):
    def __init__(self, in_dim, out_shape, out_channels=1, C=32):
        super().__init__()
        self.min_ch = C
        self.out_shape = out_shape
        self.decode_latent = FC_BN_ReLU(in_dim, C*8*out_shape[0]*out_shape[1]//256)
        self.up1 = UpCat(C*8,C*4)
        self.up2 = UpCat(C*4,C*2)
        self.up3 = UpCat(C*2,C)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            Conv_BN_ReLU(C,C, k=1),
            nn.Conv2d(C,out_channels, kernel_size=3, padding=1),
        )
    def forward(self, features):
        x1, x2, x3, z = features
        z = self.decode_latent(z)
        z = z.view(z.size(0), self.min_ch*8, self.out_shape[0]//16, self.out_shape[1]//16)
        o = self.up1(z,x3)
        o = self.up2(o,x2)
        o = self.up3(o,x1)
        return self.final(o)


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

