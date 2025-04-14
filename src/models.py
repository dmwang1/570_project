import torch
import torch.nn as nn
import torchvision.models as models

class StandardTemperatureScaling(nn.Module):
    """
    Standard Temperature Scaling calibration method.
    Uses a single temperature parameter for all classes.
    """
    def __init__(self, init_temp=1.5):
        super(StandardTemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
        
    def forward(self, logits):
        """
        Scales the logits by the temperature parameter.
        
        Args:
            logits: Logits from the base model (batch_size, num_classes)
            
        Returns:
            Calibrated logits
        """
        return logits / self.temperature
    
    def get_temperatures(self):
        """Returns the single temperature parameter."""
        return self.temperature.item()


class ClassAdaptiveCalibration(nn.Module):
    """
    Class-Adaptive Temperature Scaling calibration method.
    Uses separate temperature parameters for each class.
    """
    def __init__(self, num_classes=100, init_temp=1.5):
        super(ClassAdaptiveCalibration, self).__init__()
        # Initialize with different temperatures for each class
        # Using a parameter per class instead of a single parameter
        self.temperatures = nn.Parameter(torch.ones(num_classes) * init_temp)
        self.num_classes = num_classes
        
    def forward(self, logits):
        """
        Scales the logits by class-specific temperature parameters.
        
        Args:
            logits: Logits from the base model (batch_size, num_classes)
            
        Returns:
            Calibrated logits
        """
        # Apply class-specific temperature scaling
        # We need to scale each column (class) of the logits tensor by its corresponding temperature
        return logits / self.temperatures.unsqueeze(0)
    
    def get_temperatures(self):
        """Returns the temperature parameters for all classes."""
        return self.temperatures.detach().cpu().numpy()


def get_pretrained_model(num_classes=100):
    """
    Get a pre-trained ResNet-50 model and modify the final layer for CIFAR-100.
    
    Args:
        num_classes: Number of classes in the dataset
        
    Returns:
        Pre-trained model with modified final layer
    """
    # Load pre-trained ResNet-50
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Modify the final fully connected layer for CIFAR-100
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model