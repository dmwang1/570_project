import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from src.models import ClassAdaptiveCalibration, StandardTemperatureScaling, get_pretrained_model
from src.training import evaluate_calibration, get_calibration_results
from src.visualization import plot_reliability_diagram, plot_temperature_distribution, plot_ece_per_class

def run_demo():
    """Generate visualizations for the demo video"""
    # Set up directories
    os.makedirs('demo_results', exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    print("Loading CIFAR-100 dataset...")
    val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)
    class_names = val_set.classes
    
    # Load base model
    print("Loading pre-trained model...")
    base_model = get_pretrained_model(num_classes=100)
    base_model = base_model.to(device)
    
    # Load calibration models
    print("Loading calibration models...")
    standard_calibration = StandardTemperatureScaling()
    adaptive_calibration = ClassAdaptiveCalibration(num_classes=100)
    
    # Load saved model weights if available
    try:
        standard_calibration.load_state_dict(torch.load('models/standard_calibration.pth'))
        adaptive_calibration.load_state_dict(torch.load('models/adaptive_calibration.pth'))
        print("Loaded saved calibration models")
    except FileNotFoundError:
        print("Could not find saved models. Please run main.py first.")
        return
    
    standard_calibration = standard_calibration.to(device)
    adaptive_calibration = adaptive_calibration.to(device)
    
    # Generate results
    print("Generating demo visualizations...")
    
    # Get results for both calibration methods
    standard_results = get_calibration_results(base_model, standard_calibration, val_loader, device)
    adaptive_results = get_calibration_results(base_model, adaptive_calibration, val_loader, device)
    
    # Calculate metrics
    standard_ece, standard_ace = evaluate_calibration(base_model, standard_calibration, val_loader, device)
    adaptive_ece, adaptive_ace = evaluate_calibration(base_model, adaptive_calibration, val_loader, device)
    
    # Print results for demo
    print("\nCalibration Results:")
    print(f"Standard Temperature Scaling - ECE: {standard_ece:.4f}, ACE: {standard_ace:.4f}")
    print(f"Class-Adaptive Calibration - ECE: {adaptive_ece:.4f}, ACE: {adaptive_ace:.4f}")
    print(f"ECE Improvement: {standard_ece - adaptive_ece:.4f} ({(standard_ece - adaptive_ece) / standard_ece * 100:.2f}%)")
    print(f"ACE Improvement: {standard_ace - adaptive_ace:.4f} ({(standard_ace - adaptive_ace) / standard_ace * 100:.2f}%)")
    
    # Create demo visualizations
    
    # 1. Reliability diagrams (comparative)
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plot_reliability_diagram(
        standard_results['confidences'],
        standard_results['predictions'],
        standard_results['labels'],
        title=f"Standard Calibration (ECE: {standard_ece:.4f})"
    )
    
    plt.subplot(1, 2, 2)
    plot_reliability_diagram(
        adaptive_results['confidences'],
        adaptive_results['predictions'],
        adaptive_results['labels'],
        title=f"Class-Adaptive Calibration (ECE: {adaptive_ece:.4f})"
    )
    
    plt.tight_layout()
    plt.savefig('demo_results/reliability_comparison.png')
    plt.close()
    
    # 2. Temperature distribution across classes
    plot_temperature_distribution(
        adaptive_calibration,
        class_names=None,  # Too many classes for clear visualization
        save_path='demo_results/temperature_distribution.png'
    )
    
    # 3. ECE per class comparison
    std_ece_per_class = plot_ece_per_class(
        standard_results['confidences'],
        standard_results['predictions'],
        standard_results['labels'],
        standard_results['labels'],
        save_path='demo_results/ece_per_class_standard.png'
    )
    
    adp_ece_per_class = plot_ece_per_class(
        adaptive_results['confidences'],
        adaptive_results['predictions'],
        adaptive_results['labels'],
        adaptive_results['labels'],
        save_path='demo_results/ece_per_class_adaptive.png'
    )
    
    # 4. ECE improvement per class
    ece_improvement = np.array(std_ece_per_class) - np.array(adp_ece_per_class)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ece_improvement)), ece_improvement)
    plt.xlabel('Class Index')
    plt.ylabel('ECE Improvement')
    plt.title('ECE Improvement per Class (Standard vs Adaptive)')
    plt.tight_layout()
    plt.savefig('demo_results/ece_improvement.png')
    plt.close()
    
    # 5. Sample difficult classes (with highest initial ECE)
    difficult_classes = np.argsort(std_ece_per_class)[-5:]  # Top 5 highest ECE classes
    
    print("\nAnalysis of difficult classes (high initial ECE):")
    for cls in difficult_classes:
        print(f"Class {cls} ({class_names[cls]}):")
        print(f"  - Standard ECE: {std_ece_per_class[cls]:.4f}")
        print(f"  - Adaptive ECE: {adp_ece_per_class[cls]:.4f}")
        print(f"  - Improvement: {ece_improvement[cls]:.4f} ({ece_improvement[cls] / std_ece_per_class[cls] * 100:.2f}%)")
    
    print("\nDemo visualizations saved to demo_results/")

if __name__ == "__main__":
    run_demo()