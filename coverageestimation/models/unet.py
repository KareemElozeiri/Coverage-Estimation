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
        # in_channels: channels from the deeper layer (to be upsampled)
        # out_channels: target output channels
        # After upsampling and concatenation, we need to reduce channels
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # No channel reduction during upsampling with bilinear
            # After concat: in_channels + skip_channels -> out_channels
            # Assume skip_channels = out_channels for standard UNet symmetry
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            # With transpose conv, we reduce channels during upsampling
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # After concat: (in_channels // 2) + out_channels -> out_channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        # x1: from deeper layer (to be upsampled)
        # x2: skip connection from encoder
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection and upsampled features
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
    def __init__(self, input_channels, output_channels, bilinear=False, base_channels=32, max_depth=4):
        """
        Memory-efficient UNet with configurable depth and base channels
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels  
            bilinear: Use bilinear upsampling instead of transpose convolution
            base_channels: Base number of channels (default 32)
            max_depth: Maximum depth of the network (reduced to 4 for memory efficiency)
        """
        self.bilinear = bilinear
        self.base_channels = base_channels
        self.max_depth = max_depth
        super(UNet, self).__init__(input_channels, output_channels)

    def _create_model(self):
        # More conservative channel progression to avoid memory issues
        # Channels: [32, 64, 128, 256, 512] for max_depth=4
        channels = []
        for i in range(self.max_depth + 1):
            ch = self.base_channels * (2 ** i)
            # Cap at 512 to prevent excessive memory usage
            ch = min(ch, 512)
            channels.append(ch)
        
        print(f"Channel progression: {channels}")
        
        model_dict = nn.ModuleDict()
        
        # Input convolution
        model_dict['inc'] = DoubleConv(self.input_channels, channels[0])
        
        # Encoder (downsampling path)
        for i in range(self.max_depth):
            model_dict[f'down{i+1}'] = Down(channels[i], channels[i+1])
        
        # Decoder (upsampling path)
        # Work backwards: from deepest to shallowest
        for i in range(self.max_depth):
            # Decoder level (1-indexed)
            dec_level = i + 1
            
            # Current position in channel array (from deep to shallow)
            current_ch_idx = self.max_depth - i      # Current deep layer
            target_ch_idx = self.max_depth - i - 1   # Target shallow layer
            
            in_channels = channels[current_ch_idx]    # From deeper layer
            out_channels = channels[target_ch_idx]    # Target (matches skip connection)
            
            print(f"up{dec_level}: {in_channels} -> {out_channels} (skip: {out_channels})")
            model_dict[f'up{dec_level}'] = Up(in_channels, out_channels, self.bilinear)
        
        # Output convolution
        model_dict['outc'] = OutConv(channels[0], self.output_channels)
        
        return model_dict

    def forward(self, x):
        # Store all encoder features including input
        encoder_features = []
        
        # Initial convolution
        x1 = self.model['inc'](x)
        encoder_features.append(x1)
        
        # Encoder path - store all intermediate features
        x = x1
        for i in range(self.max_depth):
            x = self.model[f'down{i+1}'](x)
            encoder_features.append(x)
        
        # Start decoder with the deepest feature
        x = encoder_features.pop()  # Remove and use the deepest feature
        
        # Decoder path with skip connections
        for i in range(self.max_depth):
            dec_level = i + 1
            # Get the corresponding encoder feature for skip connection
            skip_connection = encoder_features.pop()  # Work backwards through encoder features
            x = self.model[f'up{dec_level}'](x, skip_connection)
        
        # Output layer
        logits = self.model['outc'](x)
        return logits

    def get_model_name(self):
        return f"UNet-{self.max_depth}Level-bc{self.base_channels}"

    def use_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        for name, module in self.model.named_children():
            if name != 'outc':  # Don't checkpoint the final output layer
                self.model[name] = torch.utils.checkpoint.checkpoint_wrapper(module)
    

