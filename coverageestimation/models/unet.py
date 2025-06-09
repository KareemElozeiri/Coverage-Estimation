import torch
import torch.nn as nn
from base_model import BaseTensorCNN

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(BaseTensorCNN):
    def __init__(self, input_channels=5, output_channels=1, base_channels=32):
        # Reduce base channels to save memory
        self.base_channels = base_channels
        super(UNet, self).__init__(input_channels, output_channels)

    def _create_model(self):
        # Use smaller channel counts to reduce memory usage
        bc = self.base_channels  # base channels
        
        return nn.ModuleDict({
            'encoder1': ConvBlock(self.input_channels, bc),
            'encoder2': ConvBlock(bc, bc*2),
            'encoder3': ConvBlock(bc*2, bc*4),
            'encoder4': ConvBlock(bc*4, bc*8),
            'pool': nn.MaxPool2d(kernel_size=2, stride=2),
            'bottleneck': ConvBlock(bc*8, bc*16),
            'upconv4': UpConv(bc*16, bc*8),
            'decoder4': ConvBlock(bc*16, bc*8),  # bc*8 from upconv + bc*8 from skip = bc*16 input
            'upconv3': UpConv(bc*8, bc*4),
            'decoder3': ConvBlock(bc*8, bc*4),   # bc*4 from upconv + bc*4 from skip = bc*8 input
            'upconv2': UpConv(bc*4, bc*2),
            'decoder2': ConvBlock(bc*4, bc*2),   # bc*2 from upconv + bc*2 from skip = bc*4 input
            'upconv1': UpConv(bc*2, bc),
            'decoder1': ConvBlock(bc*2, bc),     # bc from upconv + bc from skip = bc*2 input
            'final_conv': nn.Conv2d(bc, self.output_channels, kernel_size=1, stride=1, padding=0)
        })

    def forward(self, x):
        # Encoder
        enc1 = self.model['encoder1'](x)
        enc2 = self.model['encoder2'](self.model['pool'](enc1))
        enc3 = self.model['encoder3'](self.model['pool'](enc2))
        enc4 = self.model['encoder4'](self.model['pool'](enc3))

        # Bottleneck
        bottleneck = self.model['bottleneck'](self.model['pool'](enc4))

        # Decoder with skip connections
        dec4 = self.model['upconv4'](bottleneck)
        # Clear bottleneck from memory
        del bottleneck
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.model['decoder4'](dec4)

        dec3 = self.model['upconv3'](dec4)
        # Clear enc4 from memory
        del enc4
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.model['decoder3'](dec3)

        dec2 = self.model['upconv2'](dec3)
        # Clear enc3 from memory
        del enc3
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.model['decoder2'](dec2)

        dec1 = self.model['upconv1'](dec2)
        # Clear enc2 from memory
        del enc2
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.model['decoder1'](dec1)

        out = self.model['final_conv'](dec1)

        return out
    
    def get_model_name(self):
        return f"UNet_bc{self.base_channels}"