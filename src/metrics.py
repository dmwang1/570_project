import numpy as np

def expected_calibration_error(confidences, predictions, labels, num_bins=15):
    """Compute Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.array(confidences)
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the confidences in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            # Add to ECE
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return ece

def class_adaptive_calibration_error(confidences, predictions, labels, classes, num_bins=15):
    """Compute Class-Adaptive Calibration Error (ACE)"""
    unique_classes = np.unique(classes)
    ace = 0.0
    
    for c in unique_classes:
        # Get samples of this class
        class_indices = (classes == c)
        
        if np.sum(class_indices) > 0:
            class_conf = confidences[class_indices]
            class_pred = predictions[class_indices]
            class_labels = labels[class_indices]
            
            # Calculate ECE for this class
            class_ece = expected_calibration_error(class_conf, class_pred, class_labels, num_bins)
            
            # Weight by class frequency
            ace += class_ece * (np.sum(class_indices) / len(classes))
    
    return ace