import torch
import torch.nn as nn


class UNetLK(nn.Module):
    def __init__(self, in_chans, num_classes, first_out_chans=16):
        super().__init__()
        self.down1 = Down(in_chans, first_out_chans)
        self.down2 = Down(first_out_chans, 2 * first_out_chans)
        self.down3 = Down(2 * first_out_chans, 4 * first_out_chans)
        self.down4 = Down(4 * first_out_chans, 8 * first_out_chans)
        # self.down4 = Down(8 * first_out_chans, 16 * first_out_chans)
        # self.up1   = Up(16 * first_out_chans, 8 * first_out_chans)
        self.up1   = Up(8 * first_out_chans, 4 * first_out_chans)
        self.up2   = Up(4 * first_out_chans, 2 * first_out_chans)
        self.up3   = Up(2 * first_out_chans, first_out_chans)
        self.final = Final(in_chans, first_out_chans, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        inp = x
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x, x2)
        x  = self.up3(x, x1)
        x  = self.final(x, inp)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(4,out_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBlock(in_channels, out_channels, (7,3,3), (2,2,2), (3,1,1))
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=[2,2,2], stride=[2,2,2], bias=False),
            nn.BatchNorm3d(out_channels),
            # nn.GroupNorm(4, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x

class Final(nn.Module):
    def __init__(self, in_chans, first_out_chans, num_class):
        super().__init__()
        self.conv1 = nn.Conv3d(in_chans+first_out_chans, num_class, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(first_out_chans, first_out_chans, kernel_size=[2,2,2], stride=[2,2,2], bias=False),
            nn.BatchNorm3d(first_out_chans),
            # nn.GroupNorm(4, first_out_chans),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y,x], dim=1))
        return x



if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    from torchkeras import summary
    net = UNetLK(1, 1, 32)
    summary(net, torch.randn((2,1,1024,32,256)))