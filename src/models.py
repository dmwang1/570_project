import torch
import torch.nn as nn
import torchvision.models as models

class ClassAdaptiveCalibration(nn.Module):
    def __init__(self, num_classes, init_temp=1.0):
        super(ClassAdaptiveCalibration, self).__init__()
        # Initialize a temperature parameter for each class
        self.temperatures = nn.Parameter(torch.ones(num_classes) * init_temp)
        
    def forward(self, logits):
        # Get batch size and number of classes
        batch_size, num_classes = logits.size()
        
        # Expand temperatures to match logits shape
        temps = self.temperatures.unsqueeze(0).expand(batch_size, -1)
        
        # Apply class-specific temperature scaling
        calibrated_logits = logits / temps
        
        return calibrated_logits

class StandardTemperatureScaling(nn.Module):
    def __init__(self, init_temp=1.0):
        super(StandardTemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
        
    def forward(self, logits):
        return logits / self.temperature

def get_pretrained_model(num_classes=100):
    """Load pre-trained ResNet-50 and adjust for CIFAR-100"""
    model = models.resnet50(pretrained=True)
    # Adjust final layer for the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model