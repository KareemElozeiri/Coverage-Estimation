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
    """Upscaling then double conv with explicit channel handling"""
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # Explicit channel specification to avoid confusion
        # in_channels: from deeper layer (will be upsampled)
        # skip_channels: from skip connection
        # out_channels: desired output channels
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling: in_channels (unchanged)
            # After concatenation: in_channels + skip_channels
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After transpose conv: in_channels // 2
            # After concatenation: (in_channels // 2) + skip_channels
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from deeper layer (to be upsampled)
        # x2: skip connection from encoder
        x1 = self.up(x1)
        
        # Handle size differences with proper padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate: [batch, skip_channels + upsampled_channels, H, W]
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
    def __init__(self, input_channels, output_channels, bilinear=True, base_channels=64):
        """
        Simple UNet with fixed 4-level depth and explicit channel management
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels  
            bilinear: Use bilinear upsampling instead of transpose convolution
            base_channels: Base number of channels (64 is standard)
        """
        self.bilinear = bilinear
        self.base_channels = base_channels
        super(UNet, self).__init__(input_channels, output_channels)

    def _create_model(self):
        # Fixed 4-level UNet with corrected channel calculations
        factor = 2 if self.bilinear else 1
        
        model_dict = nn.ModuleDict()
        
        # Encoder - channels double at each level
        model_dict['inc'] = DoubleConv(self.input_channels, 64)  # 5 -> 64
        model_dict['down1'] = Down(64, 128)    # 64 -> 128
        model_dict['down2'] = Down(128, 256)   # 128 -> 256
        model_dict['down3'] = Down(256, 512)   # 256 -> 512
        model_dict['down4'] = Down(512, 1024 // factor)  # 512 -> 1024 (or 512 if bilinear)
        
        # Decoder - corrected channel calculations
        # Format: Up(from_deeper, skip_connection, output)
        if self.bilinear:
            # Bilinear upsampling preserves channels, so:
            model_dict['up1'] = Up(512, 512, 512)    # 512 + 512 -> 512
            model_dict['up2'] = Up(512, 256, 256)    # 512 + 256 -> 256
            model_dict['up3'] = Up(256, 128, 128)    # 256 + 128 -> 128
            model_dict['up4'] = Up(128, 64, 64)      # 128 + 64 -> 64
        else:
            # Transpose conv halves channels, so:
            model_dict['up1'] = Up(1024, 512, 512)   # (1024->512) + 512 -> 512
            model_dict['up2'] = Up(512, 256, 256)    # (512->256) + 256 -> 256
            model_dict['up3'] = Up(256, 128, 128)    # (256->128) + 128 -> 128
            model_dict['up4'] = Up(128, 64, 64)      # (128->64) + 64 -> 64
        
        model_dict['outc'] = OutConv(64, self.output_channels)
        
        return model_dict

    def forward(self, x):
        # Encoder with stored skip connections
        x1 = self.model['inc'](x)     # 5 -> 64
        x2 = self.model['down1'](x1)  # 64 -> 128
        x3 = self.model['down2'](x2)  # 128 -> 256
        x4 = self.model['down3'](x3)  # 256 -> 512
        x5 = self.model['down4'](x4)  # 512 -> 1024 (or 512 if bilinear)
        
        # Decoder with skip connections
        x = self.model['up1'](x5, x4)  # (1024 or 512) + 512 -> 512
        x = self.model['up2'](x, x3)   # 512 + 256 -> 256
        x = self.model['up3'](x, x2)   # 256 + 128 -> 128
        x = self.model['up4'](x, x1)   # 128 + 64 -> 64
        
        logits = self.model['outc'](x)  # 64 -> output_channels
        return logits

    def get_model_name(self):
        return f"UNet-Classic-{'Bilinear' if self.bilinear else 'TransConv'}"