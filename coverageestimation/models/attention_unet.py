from base_model import BaseTensorCNN
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class AttentionUNet(BaseTensorCNN):
    def __init__(self, in_channels=2, out_channels=1):
        super(AttentionUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _create_model(self):
        # Encoder
        encoder1 = ConvBlock(self.in_channels, 64)
        encoder2 = ConvBlock(64, 128)
        encoder3 = ConvBlock(128, 256)
        encoder4 = ConvBlock(256, 512)

        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        bottleneck = ConvBlock(512, 1024)

        # Decoder
        upconv4 = UpConv(1024, 512)
        attention4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        decoder4 = ConvBlock(1024, 512)

        upconv3 = UpConv(512, 256)
        attention3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        decoder3 = ConvBlock(512, 256)

        upconv2 = UpConv(256, 128)
        attention2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        decoder2 = ConvBlock(256, 128)

        upconv1 = UpConv(128, 64)
        attention1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        decoder1 = ConvBlock(128, 64)

        final_conv = nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1, padding=0)

        # Assemble model
        return nn.Sequential(
            encoder1,
            pool,
            encoder2,
            pool,
            encoder3,
            pool,
            encoder4,
            pool,
            bottleneck,
            upconv4,
            attention4,
            decoder4,
            upconv3,
            attention3,
            decoder3,
            upconv2,
            attention2,
            decoder2,
            upconv1,
            attention1,
            decoder1,
            final_conv
        )
