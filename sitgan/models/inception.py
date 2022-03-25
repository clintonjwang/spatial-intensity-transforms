import torch, functools
nn=torch.nn
F=nn.functional

class Inceptionv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = channels
        self.conv1 = Conv(in_channels, self.C[0])
        self.down1 = Down(self.C[0], self.C[1])
        self.down2 = Down(self.C[1], self.C[2])
        self.down3 = Down(self.C[2], self.C[3])
        self.down4 = Down(self.C[3], self.C[3])
        self.low_convs = ResConvs(num_res_units, self.C[3])
        self.up1 = Up(self.C[3], self.C[2], upsampling_type=upsampling_type)
        self.up2 = Up(self.C[2], self.C[1], upsampling_type=upsampling_type)
        self.up3 = Up(self.C[1], self.C[0], upsampling_type=upsampling_type)
        self.up4 = Up(self.C[0], self.C[0], upsampling_type=upsampling_type)
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


class Conv(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.layers(x)

class ResConv(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(size, size//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(size//2),
            nn.ReLU(inplace=False),
            nn.Conv2d(size//2, size, kernel_size=1, padding=0),
            nn.BatchNorm2d(size),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return x + self.layers(x)

class ResConvs(nn.Module):
    def __init__(self, num_res_units, size):
        super().__init__()
        layers = [ResConv(size)] * num_res_units
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class Down(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2), Conv(in_size, out_size)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_size, out_size, upsampling_type="bilinear"):
        super().__init__()
        if upsampling_type == "bilinear":
            self.up = functools.partial(F.interpolate, scale_factor=2., mode="bilinear")
        else:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)#, padding=1)
        self.conv = Conv(in_size * 2, out_size)

    def forward(self, x1, x2):
        up = self.up(x1)
        return self.conv(torch.cat([up, x2], dim=1))
