import torch
import torch.nn as nn
import torchvision.models as models

class ClassAdaptiveCalibration(nn.Module):
    def __init__(self, num_classes, init_temp=1.5):
        super(ClassAdaptiveCalibration, self).__init__()
        # Initialize a temperature parameter for each class
        self.temperatures = nn.Parameter(torch.ones(num_classes) * init_temp)
        self.num_classes = num_classes
        
    def forward(self, logits):
        # Get batch size and number of classes
        batch_size, num_classes = logits.size()
        assert num_classes == self.num_classes, f"Expected {self.num_classes} classes but got {num_classes}"
        
        # More efficient vectorized implementation
        # Expand temperatures to shape [1, num_classes] for broadcasting
        temps = self.temperatures.unsqueeze(0)
        
        # Apply temperature scaling to all logits at once
        # This divides each logit by its corresponding class temperature
        scaled_logits = logits / temps
        
        return scaled_logits

class StandardTemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.5):
        super(StandardTemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
        
    def forward(self, logits):
        return logits / self.temperature

def get_pretrained_model(num_classes=100):
    """Load pre-trained ResNet-50 and adjust for CIFAR-100"""
    model = models.resnet50(weights='IMAGENET1K_V1')  # updated from pretrained=True
    # Adjust final layer for the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model