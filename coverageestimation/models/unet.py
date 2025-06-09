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
        # in_channels is the number of channels from the previous layer
        # We need to account for concatenation with skip connection
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, we concatenate with skip connection, so input to conv is in_channels + skip_channels
            # For simplicity, assume skip_channels = out_channels (this works for standard UNet)
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After transpose conv and concatenation: (in_channels // 2) + out_channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

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
    def __init__(self, input_channels, output_channels, bilinear=False, base_channels=32, max_depth=5):
        """
        Memory-efficient UNet with configurable depth and base channels
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels  
            bilinear: Use bilinear upsampling instead of transpose convolution
            base_channels: Base number of channels (default 32 instead of 64)
            max_depth: Maximum depth of the network (default 5 instead of 8)
        """
        self.bilinear = bilinear
        self.base_channels = base_channels
        self.max_depth = max_depth
        super(UNet, self).__init__(input_channels, output_channels)

    def _create_model(self):
        # Define the channel progression
        channels = [self.base_channels * (2 ** i) for i in range(self.max_depth + 1)]
        # Cap maximum channels to prevent excessive memory usage
        channels = [min(ch, 1024) for ch in channels]
        
        model_dict = {
            # Input convolution
            'inc': DoubleConv(self.input_channels, channels[0]),
        }
        
        # Encoder (downsampling path)
        for i in range(self.max_depth):
            model_dict[f'down{i+1}'] = Down(channels[i], channels[i+1])
        
        # Decoder (upsampling path)
        for i in range(self.max_depth):
            # Calculate indices for decoder
            decoder_idx = i + 1
            encoder_level = self.max_depth - i  # Which encoder level we're at
            skip_level = encoder_level - 1      # Which encoder level provides skip connection
            
            in_channels = channels[encoder_level]   # Channels from deeper layer
            out_channels = channels[skip_level]     # Target channels (matching skip connection)
            
            model_dict[f'up{decoder_idx}'] = Up(in_channels, out_channels, self.bilinear)
        
        # Output convolution
        model_dict['outc'] = OutConv(channels[0], self.output_channels)
        
        return nn.ModuleDict(model_dict)

    def forward(self, x):
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder path
        x = self.model['inc'](x)
        encoder_features.append(x)
        
        for i in range(self.max_depth):
            x = self.model[f'down{i+1}'](x)
            encoder_features.append(x)
        
        # Take the deepest feature as starting point for decoder
        x = encoder_features[-1]
        
        # Decoder path with skip connections
        for i in range(self.max_depth):
            decoder_idx = i + 1
            skip_idx = self.max_depth - 1 - i  # Index for skip connection
            
            if skip_idx >= 0:
                skip_connection = encoder_features[skip_idx]
                x = self.model[f'up{decoder_idx}'](x, skip_connection)
        
        # Output
        logits = self.model['outc'](x)
        return logits

    def get_model_name(self):
        return f"UNet-{self.max_depth}Level-bc{self.base_channels}"

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory for deep networks"""
        self.model['inc'] = torch.utils.checkpoint.checkpoint_wrapper(self.model['inc'])
        for i in range(self.max_depth):
            if f'down{i+1}' in self.model:
                self.model[f'down{i+1}'] = torch.utils.checkpoint.checkpoint_wrapper(self.model[f'down{i+1}'])
            if f'up{i+1}' in self.model:
                self.model[f'up{i+1}'] = torch.utils.checkpoint.checkpoint_wrapper(self.model[f'up{i+1}'])
    
    def print_architecture(self):
        """Print the architecture for debugging"""
        channels = [self.base_channels * (2 ** i) for i in range(self.max_depth + 1)]
        channels = [min(ch, 1024) for ch in channels]
        
        print(f"UNet Architecture (depth={self.max_depth}, base_channels={self.base_channels}):")
        print(f"Input channels: {self.input_channels}")
        print(f"Channel progression: {channels}")
        
        print("\nEncoder:")
        print(f"  inc: {self.input_channels} -> {channels[0]}")
        for i in range(self.max_depth):
            print(f"  down{i+1}: {channels[i]} -> {channels[i+1]}")
        
        print("\nDecoder:")
        for i in range(self.max_depth):
            decoder_idx = i + 1
            encoder_level = self.max_depth - i
            skip_level = encoder_level - 1
            in_ch = channels[encoder_level]
            out_ch = channels[skip_level]
            concat_ch = in_ch + out_ch if not self.bilinear else in_ch + out_ch
            print(f"  up{decoder_idx}: {in_ch} + {out_ch} (skip) -> {concat_ch} -> {out_ch}")
        
        print(f"  outc: {channels[0]} -> {self.output_channels}")