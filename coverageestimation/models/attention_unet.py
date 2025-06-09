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
    def __init__(self, input_channels=2, output_channels=1, base_channels=32):
        # Reduce base channels to save memory
        self.base_channels = base_channels
        super(AttentionUNet, self).__init__(input_channels, output_channels)

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
            'attention4': AttentionBlock(F_g=bc*8, F_l=bc*8, F_int=bc*4),
            'decoder4': ConvBlock(bc*16, bc*8),
            'upconv3': UpConv(bc*8, bc*4),
            'attention3': AttentionBlock(F_g=bc*4, F_l=bc*4, F_int=bc*2),
            'decoder3': ConvBlock(bc*8, bc*4),
            'upconv2': UpConv(bc*4, bc*2),
            'attention2': AttentionBlock(F_g=bc*2, F_l=bc*2, F_int=bc),
            'decoder2': ConvBlock(bc*4, bc*2),
            'upconv1': UpConv(bc*2, bc),
            'attention1': AttentionBlock(F_g=bc, F_l=bc, F_int=bc//2),
            'decoder1': ConvBlock(bc*2, bc),
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

        # Decoder with gradient checkpointing for memory efficiency
        dec4 = self.model['upconv4'](bottleneck)
        # Clear bottleneck from memory
        del bottleneck
        dec4 = torch.cat((self.model['attention4'](dec4, enc4), dec4), dim=1)
        dec4 = self.model['decoder4'](dec4)

        dec3 = self.model['upconv3'](dec4)
        # Clear enc4 from memory
        del enc4
        dec3 = torch.cat((self.model['attention3'](dec3, enc3), dec3), dim=1)
        dec3 = self.model['decoder3'](dec3)

        dec2 = self.model['upconv2'](dec3)
        # Clear enc3 from memory
        del enc3
        dec2 = torch.cat((self.model['attention2'](dec2, enc2), dec2), dim=1)
        dec2 = self.model['decoder2'](dec2)

        dec1 = self.model['upconv1'](dec2)
        # Clear enc2 from memory
        del enc2
        dec1 = torch.cat((self.model['attention1'](dec1, enc1), dec1), dim=1)
        dec1 = self.model['decoder1'](dec1)

        out = self.model['final_conv'](dec1)

        return out
    
    def get_model_name(self):
        return f"AttentionUNet_bc{self.base_channels}"