import numpy as np
import matplotlib.pyplot as plt
from .metrics import expected_calibration_error

def plot_reliability_diagram(confidences, predictions, targets, title=None, save_path=None, n_bins=15):
    """
    Plot reliability diagram.
    
    Args:
        confidences: Predicted confidence scores
        predictions: Predicted class labels
        targets: True class labels
        title: Optional title for the plot
        save_path: Optional path to save the figure
        n_bins: Number of bins for confidence scores
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    # Initialize arrays for bin accuracies and confidences
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Calculate accuracy and average confidence for each bin
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_counts[i] = np.sum(in_bin)
        
        if bin_counts[i] > 0:
            # Calculate accuracy and average confidence
            bin_accuracies[i] = np.mean(predictions[in_bin] == targets[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
    
    # Calculate ECE
    ece = expected_calibration_error(confidences, predictions, targets, n_bins)
    
    # Plot reliability diagram
    plt.figure(figsize=(8, 6))
    
    # Plot the bin confidences and accuracies
    bin_sizes_scaled = bin_counts / np.max(bin_counts) * 0.8  # Scale for visualization
    plt.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.8, edgecolor='black', label='Accuracy')
    plt.bar(bin_centers, bin_confidences, width=1/n_bins, alpha=0.6, edgecolor='red', label='Confidence', color='red')
    
    # Plot the diagonal (perfect calibration)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot histogram of confidence distribution
    plt.bar(bin_centers, bin_sizes_scaled, width=1/n_bins, alpha=0.3, edgecolor='gray', label='Data')
    
    # Add labels and title
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy / Count (scaled)')
    if title:
        plt.title(f"{title} (ECE: {ece:.4f})")
    else:
        plt.title(f"Reliability Diagram (ECE: {ece:.4f})")
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()


def plot_temperature_distribution(model, class_names=None, save_path=None):
    """
    Plot temperature distribution across classes.
    
    Args:
        model: Calibration model
        class_names: Optional list of class names
        save_path: Optional path to save the figure
    """
    if not hasattr(model, 'temperatures'):
        print("Model does not have multiple temperature parameters.")
        return
    
    # Get temperature values
    temperatures = model.get_temperatures()
    
    # Calculate statistics
    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    mean_temp = np.mean(temperatures)
    std_temp = np.std(temperatures)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot temperature distribution
    class_indices = np.arange(len(temperatures))
    plt.bar(class_indices, temperatures, alpha=0.7)
    
    # Add reference line for mean temperature
    plt.axhline(y=mean_temp, color='r', linestyle='-', label=f'Mean: {mean_temp:.4f}')
    plt.axhline(y=1.0, color='k', linestyle='--', label='T=1 (No calibration)')
    
    # Add labels and title
    plt.xlabel('Class Index')
    plt.ylabel('Temperature Parameter')
    plt.title(f'Temperature Distribution Across Classes (Mean: {mean_temp:.4f}, Std: {std_temp:.4f})')
    
    # Add class names if provided (but limit to avoid clutter)
    if class_names and len(class_names) <= 20:
        plt.xticks(class_indices, class_names, rotation=90)
    elif len(temperatures) > 20:
        # Only show some ticks for readability
        plt.xticks(np.arange(0, len(temperatures), len(temperatures)//10))
    
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()


def plot_ece_per_class(confidences, predictions, targets, class_indices, n_bins=15, save_path=None):
    """
    Calculate and plot ECE for each class.
    
    Args:
        confidences: Predicted confidence scores
        predictions: Predicted class labels
        targets: True class labels
        class_indices: List or array of class indices
        n_bins: Number of bins for confidence scores
        save_path: Optional path to save the figure
        
    Returns:
        List of ECE values per class
    """
    unique_classes = np.unique(class_indices)
    num_classes = len(unique_classes)
    
    # Initialize arrays
    class_ece = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    
    # Calculate ECE for each class
    for i, c in enumerate(unique_classes):
        # Find samples predicted as this class
        class_predictions = predictions == c
        class_counts[i] = np.sum(class_predictions)
        
        if class_counts[i] > 0:
            # Get confidences and true labels for these predictions
            class_confidences = confidences[class_predictions]
            class_true_labels = (targets[class_predictions] == c).astype(int)
            class_pred_labels = np.ones_like(class_true_labels)  # All predictions are for this class
            
            # Calculate ECE for this class
            class_ece[i] = expected_calibration_error(
                class_confidences, class_pred_labels, class_true_labels, n_bins)
    
    # Only plot if we have a reasonable number of classes
    if num_classes <= 100:  # Adjust this threshold as needed
        plt.figure(figsize=(12, 6))
        
        # Sort classes by ECE for better visualization
        sort_indices = np.argsort(class_ece)[::-1]  # Descending order
        sorted_classes = unique_classes[sort_indices]
        sorted_ece = class_ece[sort_indices]
        sorted_counts = class_counts[sort_indices]
        
        # Normalize counts for visualization
        normalized_counts = sorted_counts / np.max(sorted_counts) * 0.5
        
        # Plot ECE per class
        bars = plt.bar(np.arange(num_classes), sorted_ece, alpha=0.7)
        
        # Add count information as smaller bars
        plt.bar(np.arange(num_classes), normalized_counts, alpha=0.3, color='gray', label='Relative Frequency')
        
        # Add mean ECE line
        mean_ece = np.mean(class_ece)
        plt.axhline(y=mean_ece, color='r', linestyle='-', 
                   label=f'Mean ECE: {mean_ece:.4f}')
        
        # Add labels and title
        plt.xlabel('Class Index (sorted by ECE)')
        plt.ylabel('Expected Calibration Error')
        plt.title(f'ECE per Class (Mean: {mean_ece:.4f}, Max: {np.max(class_ece):.4f})')
        
        # Show only a subset of x-ticks for readability
        if num_classes > 20:
            plt.xticks(np.arange(0, num_classes, num_classes//10))
        
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
    
    return class_ece.tolist()