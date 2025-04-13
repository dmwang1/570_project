import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .metrics import expected_calibration_error, class_adaptive_calibration_error

def train_calibration(base_model, calibration_model, train_loader, val_loader, lr=0.01, epochs=50, 
                     l2_reg_strength=0.01, device=None):
    """Train the calibration model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move models to device
    base_model = base_model.to(device)
    calibration_model = calibration_model.to(device)
    
    # Freeze base model weights
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Use NLL loss for calibration
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(calibration_model.parameters(), lr=lr)
    
    # Keep track of best model
    best_ace = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        calibration_model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get logits from base model
            with torch.no_grad():
                logits = base_model(inputs)
            
            # Apply calibration
            calibrated_logits = calibration_model(logits)
            
            # Calculate loss
            loss = criterion(calibrated_logits, labels)
            
            # Apply L2 regularization on temperature parameters
            if hasattr(calibration_model, 'temperatures'):
                l2_reg = l2_reg_strength * torch.norm(calibration_model.temperatures)
                loss += l2_reg
            elif hasattr(calibration_model, 'temperature'):
                l2_reg = l2_reg_strength * torch.norm(calibration_model.temperature)
                loss += l2_reg
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            ece, ace = evaluate_calibration(base_model, calibration_model, val_loader, device)
            print(f"Validation - ECE: {ece:.4f}, ACE: {ace:.4f}")
            
            if ace < best_ace:
                best_ace = ace
                best_model_state = calibration_model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        calibration_model.load_state_dict(best_model_state)
    
    return calibration_model

def evaluate_calibration(base_model, calibration_model, data_loader, device):
    """Evaluate calibration performance"""
    base_model.eval()
    calibration_model.eval()
    
    all_confidences = []
    all_predictions = []
    all_labels = []
    all_classes = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get logits from base model
            logits = base_model(inputs)
            
            # Apply calibration
            calibrated_logits = calibration_model(logits)
            probas = torch.softmax(calibrated_logits, dim=1)
            
            # Get confidence and predictions
            confidences, predictions = torch.max(probas, dim=1)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_classes.extend(labels.cpu().numpy())  # For ACE
    
    # Calculate metrics
    all_confidences = np.array(all_confidences)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_classes = np.array(all_classes)
    
    ece = expected_calibration_error(all_confidences, all_predictions, all_labels)
    ace = class_adaptive_calibration_error(all_confidences, all_predictions, all_labels, all_classes)
    
    return ece, ace

def get_calibration_results(base_model, calibration_model, data_loader, device):
    """Get detailed results for visualization"""
    base_model.eval()
    calibration_model.eval()
    
    all_confidences = []
    all_predictions = []
    all_labels = []
    all_logits = []
    all_calibrated_logits = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get logits from base model
            logits = base_model(inputs)
            
            # Apply calibration
            calibrated_logits = calibration_model(logits)
            probas = torch.softmax(calibrated_logits, dim=1)
            
            # Get confidence and predictions
            confidences, predictions = torch.max(probas, dim=1)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_calibrated_logits.append(calibrated_logits.cpu().numpy())
    
    return {
        'confidences': np.array(all_confidences),
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'logits': np.concatenate(all_logits) if all_logits else np.array([]),
        'calibrated_logits': np.concatenate(all_calibrated_logits) if all_calibrated_logits else np.array([])
    }