import matplotlib.pyplot as plt
import numpy as np
import os

def plot_reliability_diagram(confidences, predictions, labels, num_bins=15, title="Reliability Diagram", save_path=None):
    """Plot reliability diagram to visually demonstrate calibration"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences_in_bin = []
    bin_sizes = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        bin_sizes.append(bin_size)
        
        if bin_size > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            accuracies.append(accuracy_in_bin)
            confidences_in_bin.append(avg_confidence_in_bin)
        else:
            accuracies.append(0)
            confidences_in_bin.append(0)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.bar(bin_lowers, accuracies, width=(bin_uppers - bin_lowers), alpha=0.3, label='Outputs')
    plt.plot(confidences_in_bin, accuracies, 'ro-', label='Accuracy')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.close()

def plot_temperature_distribution(calibration_model, class_names=None, save_path=None):
    """Plot the distribution of temperature parameters across classes"""
    if not hasattr(calibration_model, 'temperatures'):
        print("Model does not have class-specific temperatures")
        return
    
    temperatures = calibration_model.temperatures.detach().cpu().numpy()
    num_classes = len(temperatures)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), temperatures)
    plt.xlabel('Class Index')
    plt.ylabel('Temperature Value')
    plt.title('Temperature Parameter Distribution Across Classes')
    
    if class_names and len(class_names) == num_classes:
        plt.xticks(range(num_classes), class_names, rotation=90)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.close()

def plot_ece_per_class(confidences, predictions, labels, classes, num_bins=15, save_path=None):
    """Plot ECE for each class"""
    unique_classes = np.unique(classes)
    ece_per_class = []
    
    for c in unique_classes:
        class_indices = (classes == c)
        
        if np.sum(class_indices) > 0:
            class_conf = confidences[class_indices]
            class_pred = predictions[class_indices]
            class_labels = labels[class_indices]
            
            from .metrics import expected_calibration_error
            class_ece = expected_calibration_error(class_conf, class_pred, class_labels, num_bins)
            ece_per_class.append(class_ece)
        else:
            ece_per_class.append(0)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique_classes)), ece_per_class)
    plt.xlabel('Class Index')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('ECE per Class')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.close()
    
    return ece_per_class