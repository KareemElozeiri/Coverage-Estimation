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

class UNet8Level(BaseTensorCNN):
    def __init__(self, input_channels, output_channels, bilinear=False):
        self.bilinear = bilinear
        super(UNet8Level, self).__init__(input_channels, output_channels)

    def _create_model(self):
        # Define the channel progression for 8 levels
        # Start with 64 channels and double at each level
        channels = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
        # Create a sequential container for the U-Net components
        model = nn.ModuleDict({
            # Input convolution
            'inc': DoubleConv(self.input_channels, channels[0]),
            
            # Encoder (downsampling path) - 7 down blocks for 8 levels total
            'down1': Down(channels[0], channels[1]),
            'down2': Down(channels[1], channels[2]),
            'down3': Down(channels[2], channels[3]),
            'down4': Down(channels[3], channels[4]),
            'down5': Down(channels[4], channels[5]),
            'down6': Down(channels[5], channels[6]),
            'down7': Down(channels[6], channels[7]),
            
            # Decoder (upsampling path) - 7 up blocks
            'up1': Up(channels[7], channels[6] // (2 if self.bilinear else 1), self.bilinear),
            'up2': Up(channels[6], channels[5] // (2 if self.bilinear else 1), self.bilinear),
            'up3': Up(channels[5], channels[4] // (2 if self.bilinear else 1), self.bilinear),
            'up4': Up(channels[4], channels[3] // (2 if self.bilinear else 1), self.bilinear),
            'up5': Up(channels[3], channels[2] // (2 if self.bilinear else 1), self.bilinear),
            'up6': Up(channels[2], channels[1] // (2 if self.bilinear else 1), self.bilinear),
            'up7': Up(channels[1], channels[0], self.bilinear),
            
            # Output convolution
            'outc': OutConv(channels[0], self.output_channels)
        })
        
        return model

    def forward(self, x):
        # Encoder path
        x1 = self.model['inc'](x)
        x2 = self.model['down1'](x1)
        x3 = self.model['down2'](x2)
        x4 = self.model['down3'](x3)
        x5 = self.model['down4'](x4)
        x6 = self.model['down5'](x5)
        x7 = self.model['down6'](x6)
        x8 = self.model['down7'](x7)
        
        # Decoder path with skip connections
        x = self.model['up1'](x8, x7)
        x = self.model['up2'](x, x6)
        x = self.model['up3'](x, x5)
        x = self.model['up4'](x, x4)
        x = self.model['up5'](x, x3)
        x = self.model['up6'](x, x2)
        x = self.model['up7'](x, x1)
        
        # Output
        logits = self.model['outc'](x)
        return logits

    def get_model_name(self):
        return "UNet-8Level-TensorToTensor"

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory for deep networks"""
        self.model['inc'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['inc'])
        self.model['down1'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['down1'])
        self.model['down2'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['down2'])
        self.model['down3'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['down3'])
        self.model['down4'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['down4'])
        self.model['down5'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['down5'])
        self.model['down6'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['down6'])
        self.model['down7'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['down7'])