import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import BaseTensorCNN

class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(BaseTensorCNN):
    def __init__(self, input_channels, output_channels, bilinear=False):
        self.bilinear = bilinear
        super(UNet, self).__init__(input_channels, output_channels)

    def _create_model(self):
        # Define the channel progression for 8 levels
        # Start with 64 channels and double at each level
        channels = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
        # Input convolution
        self.inc = DoubleConv(self.input_channels, channels[0])
        
        # Encoder (downsampling path) - 7 down blocks for 8 levels total
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4])
        self.down5 = Down(channels[4], channels[5])
        self.down6 = Down(channels[5], channels[6])
        self.down7 = Down(channels[6], channels[7])
        
        # Decoder (upsampling path) - 7 up blocks
        factor = 2 if self.bilinear else 1
        self.up1 = Up(channels[7], channels[6] // factor, self.bilinear)
        self.up2 = Up(channels[6], channels[5] // factor, self.bilinear)
        self.up3 = Up(channels[5], channels[4] // factor, self.bilinear)
        self.up4 = Up(channels[4], channels[3] // factor, self.bilinear)
        self.up5 = Up(channels[3], channels[2] // factor, self.bilinear)
        self.up6 = Up(channels[2], channels[1] // factor, self.bilinear)
        self.up7 = Up(channels[1], channels[0], self.bilinear)
        
        # Output convolution
        self.outc = OutConv(channels[0], self.output_channels)
        
        # Return self to work with the base class structure
        return self

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        
        # Decoder path with skip connections
        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits

    def get_model_name(self):
        return "UNet-8Level-TensorToTensor"

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory for deep networks"""
        self.inc = torch.utils.checkpoint.checkpoint_wrapper(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint_wrapper(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint_wrapper(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint_wrapper(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint_wrapper(self.down4)
        self.down5 = torch.utils.checkpoint.checkpoint_wrapper(self.down5)
        self.down6 = torch.utils.checkpoint.checkpoint_wrapper(self.down6)
        self.down7 = torch.utils.checkpoint.checkpoint_wrapper(self.down7)