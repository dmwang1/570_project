import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

from .metrics import expected_calibration_error, class_adaptive_calibration_error

def train_calibration(base_model, calibration_model, train_loader, val_loader, 
                      lr=0.01, epochs=50, l2_reg_strength=0.01, device='cuda', patience=5):
    """
    Train a calibration model using NLL loss with L2 regularization.
    
    Args:
        base_model: Pre-trained classification model
        calibration_model: Calibration model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        lr: Learning rate
        epochs: Number of training epochs
        l2_reg_strength: L2 regularization strength
        device: Device to use for training
        patience: Early stopping patience
        
    Returns:
        Trained calibration model
    """
    base_model.eval()  # Set base model to evaluation mode
    calibration_model.to(device)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # For class-adaptive, use different learning rate and regularization
    if hasattr(calibration_model, 'temperatures') and calibration_model.temperatures.size(0) > 1:
        print(f"Class-Adaptive model: using L2 reg={l2_reg_strength}, lr={lr/2}")
        optimizer = optim.Adam(calibration_model.parameters(), lr=lr/2)
    else:
        print(f"Standard model: using L2 reg={l2_reg_strength}, lr={lr}")
        optimizer = optim.Adam(calibration_model.parameters(), lr=lr)
    
    # Variables for early stopping
    best_ece = float('inf')
    best_ace = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        calibration_model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get logits from base model (without gradients)
            with torch.no_grad():
                logits = base_model(inputs)
            
            # Apply calibration
            calibrated_logits = calibration_model(logits)
            
            # Calculate NLL loss
            nll_loss = criterion(calibrated_logits, targets)
            
            # Add L2 regularization
            if hasattr(calibration_model, 'temperature'):
                # Standard temperature scaling - regularize single parameter
                l2_reg = l2_reg_strength * (calibration_model.temperature - 1.0) ** 2
            else:
                # Class-adaptive temperature scaling - regularize all parameters
                # Encourage temperatures to be close to 1 to prevent overfitting
                l2_reg = l2_reg_strength * torch.mean((calibration_model.temperatures - 1.0) ** 2)
            
            # Total loss
            loss = nll_loss + l2_reg
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Validation phase
        calibration_model.eval()
        val_ece, val_ace = evaluate_calibration(base_model, calibration_model, val_loader, device)
        print(f"Validation - ECE: {val_ece:.4f}, ACE: {val_ace:.4f}")
        
        # Check if this is the best model so far
        if val_ece < best_ece:
            best_ece = val_ece
            best_ace = val_ace
            best_model_state = calibration_model.state_dict().copy()
            print(f"New best model! ECE: {best_ece:.4f}, ACE: {best_ace:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load the best model state
    if best_model_state is not None:
        calibration_model.load_state_dict(best_model_state)
        print(f"Loaded best model with ECE: {best_ece:.4f}, ACE: {best_ace:.4f}")
    
    # For class-adaptive, print temperature statistics
    if hasattr(calibration_model, 'temperatures') and calibration_model.temperatures.size(0) > 1:
        temps = calibration_model.get_temperatures()
        print(f"Temperature stats - Min: {np.min(temps):.4f}, Max: {np.max(temps):.4f}, Mean: {np.mean(temps):.4f}")
    
    return calibration_model


def evaluate_calibration(base_model, calibration_model, data_loader, device):
    """
    Evaluate calibration metrics (ECE and ACE).
    
    Args:
        base_model: Base classification model
        calibration_model: Calibration model
        data_loader: DataLoader for evaluation
        device: Device to use for evaluation
        
    Returns:
        Tuple of (ECE, ACE) metrics
    """
    base_model.eval()
    calibration_model.eval()
    
    all_confidences = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get logits from base model
            logits = base_model(inputs)
            
            # Apply calibration
            calibrated_logits = calibration_model(logits)
            
            # Get softmax probabilities
            probs = torch.softmax(calibrated_logits, dim=1)
            
            # Get predictions and confidences
            confidences, predictions = torch.max(probs, dim=1)
            
            # Store results
            all_confidences.append(confidences.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate results
    confidences = np.concatenate(all_confidences)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Calculate ECE and ACE
    ece = expected_calibration_error(confidences, predictions, targets)
    ace = class_adaptive_calibration_error(confidences, predictions, targets)
    
    return ece, ace


def get_calibration_results(base_model, calibration_model, data_loader, device):
    """
    Get detailed calibration results for analysis and visualization.
    
    Args:
        base_model: Base classification model
        calibration_model: Calibration model
        data_loader: DataLoader for evaluation
        device: Device to use for evaluation
        
    Returns:
        Dictionary containing confidences, predictions, and true labels
    """
    base_model.eval()
    calibration_model.eval()
    
    all_confidences = []
    all_predictions = []
    all_targets = []
    all_probs = []  # Store all probability distributions
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get logits from base model
            logits = base_model(inputs)
            
            # Apply calibration
            calibrated_logits = calibration_model(logits)
            
            # Get softmax probabilities
            probs = torch.softmax(calibrated_logits, dim=1)
            
            # Get predictions and confidences
            confidences, predictions = torch.max(probs, dim=1)
            
            # Store results
            all_confidences.append(confidences.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate results
    results = {
        'confidences': np.concatenate(all_confidences),
        'predictions': np.concatenate(all_predictions),
        'labels': np.concatenate(all_targets),
        'probabilities': np.concatenate(all_probs)
    }
    
    return results