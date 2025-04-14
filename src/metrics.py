import numpy as np
from sklearn.metrics import accuracy_score

def expected_calibration_error(confidences, predictions, targets, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        confidences: Predicted confidence scores
        predictions: Predicted class labels
        targets: True class labels
        n_bins: Number of bins for confidence scores
        
    Returns:
        Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Initialize counters
    ece = 0.0
    total_samples = len(confidences)
    
    # Calculate accuracy and average confidence for each bin
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Calculate accuracy and average confidence
            bin_accuracy = np.mean(predictions[in_bin] == targets[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            # Update ECE
            ece += (bin_size / total_samples) * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def class_adaptive_calibration_error(confidences, predictions, targets, n_bins=15):
    """
    Calculate Adaptive Calibration Error (ACE).
    This is a class-adaptive version of ECE that calculates calibration error per class.
    
    Args:
        confidences: Predicted confidence scores
        predictions: Predicted class labels
        targets: True class labels
        n_bins: Number of bins for confidence scores
        
    Returns:
        Adaptive Calibration Error
    """
    # Get unique classes
    unique_classes = np.unique(targets)
    n_classes = len(unique_classes)
    
    # Initialize ACE
    ace = 0.0
    
    # Calculate ECE for each class and take the average
    for c in unique_classes:
        # Find samples predicted as this class
        class_predictions = predictions == c
        
        if np.sum(class_predictions) > 0:
            # Calculate ECE for this class
            class_confidences = confidences[class_predictions]
            class_targets = (targets[class_predictions] == c).astype(int)
            class_predictions = np.ones_like(class_targets)  # All predictions are for this class
            
            # Calculate class-specific ECE
            class_ece = expected_calibration_error(class_confidences, class_predictions, class_targets, n_bins)
            
            # Weight by class frequency
            class_weight = np.sum(class_predictions) / len(predictions)
            ace += class_weight * class_ece
    
    return ace


def accuracy(predictions, targets):
    """
    Calculate accuracy.
    
    Args:
        predictions: Predicted class labels
        targets: True class labels
        
    Returns:
        Classification accuracy
    """
    return accuracy_score(targets, predictions)