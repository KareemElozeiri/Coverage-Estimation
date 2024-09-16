import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseTensorCNN(nn.Module, ABC):
    def __init__(self, input_channels, output_channels):
        super(BaseTensorCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.model = self._create_model()
    
    @abstractmethod
    def _create_model(self):
        """
        Create the entire model architecture.
        This method should be implemented by subclasses.
        """
        pass
    
    def forward(self, x):
        return self.model(x)
    
    @abstractmethod
    def get_model_name(self):
        """
        Return the name of the model.
        This method should be implemented by subclasses.
        """
        pass

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))