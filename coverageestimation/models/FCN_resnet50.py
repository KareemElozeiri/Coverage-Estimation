import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from base_model import BaseTensorCNN

class FCNResNet50TensorToTensor(BaseTensorCNN):
    def __init__(self, input_channels, output_channels):
        super(FCNResNet50TensorToTensor, self).__init__(input_channels, output_channels)

    def _create_model(self):
        # Load pretrained FCN-ResNet50
        fcn = models.fcn_resnet50(pretrained=True)
        
        # Modify the first convolutional layer if input_channels != 3
        if self.input_channels != 3:
            original_conv = fcn.backbone.conv1
            fcn.backbone.conv1 = nn.Conv2d(
                self.input_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=False
            )
        
        # Modify the final layer to match output_channels
        fcn.classifier[4] = nn.Conv2d(512, self.output_channels, kernel_size=1)
        
        # Remove auxiliary classifier if present
        if hasattr(fcn, 'aux_classifier'):
            fcn.aux_classifier = None
        
        return fcn

    def get_model_name(self):
        return "FCN-ResNet50-TensorToTensor"

    def forward(self, x):
        result = self.model(x)
        return result['out'] 