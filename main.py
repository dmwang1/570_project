import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from datetime import datetime

from src.models import ClassAdaptiveCalibration, StandardTemperatureScaling, get_pretrained_model
from src.training import train_calibration, evaluate_calibration, get_calibration_results
from src.visualization import plot_reliability_diagram, plot_temperature_distribution, plot_ece_per_class

def main(args):
    # Set up directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Set up dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    print("Loading CIFAR-100 dataset...")
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Get class names
    class_names = train_set.classes
    
    # Load base model
    print("Loading pre-trained model...")
    base_model = get_pretrained_model(num_classes=100)
    base_model = base_model.to(device)
    
    # Train base model if needed (or load pre-trained)
    if args.train_base:
        print("Training base model...")
        # Add base model training code here if needed
    
    # Initialize calibration models - using different settings for each
    print("Initializing calibration models...")
    standard_calibration = StandardTemperatureScaling(init_temp=1.5)
    
    # For class-adaptive, initialize temperatures closer to 1
    adaptive_calibration = ClassAdaptiveCalibration(num_classes=100, init_temp=1.0)
    
    # Train standard calibration first
    print("\nTraining standard temperature scaling...")
    standard_calibration = train_calibration(
        base_model, 
        standard_calibration, 
        train_loader, 
        val_loader, 
        lr=args.learning_rate,
        epochs=args.epochs,
        l2_reg_strength=args.l2_reg,
        device=device,
        patience=args.patience
    )
    
    # Save model
    torch.save(standard_calibration.state_dict(), os.path.join('models', 'standard_calibration.pth'))
    
    # Train adaptive calibration with stronger regularization
    print("\nTraining class-adaptive calibration...")
    adaptive_calibration = train_calibration(
        base_model, 
        adaptive_calibration, 
        train_loader, 
        val_loader, 
        lr=args.learning_rate,
        epochs=args.epochs * 2,  # Double the epochs for class-adaptive
        l2_reg_strength=args.l2_reg * 5,  # Stronger regularization
        device=device,
        patience=args.patience * 2
    )
    
    # Save model
    torch.save(adaptive_calibration.state_dict(), os.path.join('models', 'adaptive_calibration.pth'))
    
    # Evaluate and compare
    print("\nEvaluating calibration methods...")
    
    # Standard calibration results
    standard_results = get_calibration_results(base_model, standard_calibration, val_loader, device)
    standard_ece, standard_ace = evaluate_calibration(base_model, standard_calibration, val_loader, device)
    
    # Adaptive calibration results
    adaptive_results = get_calibration_results(base_model, adaptive_calibration, val_loader, device)
    adaptive_ece, adaptive_ace = evaluate_calibration(base_model, adaptive_calibration, val_loader, device)
    
    # Print results
    print("\nResults Summary:")
    print(f"Standard Temperature Scaling - ECE: {standard_ece:.4f}, ACE: {standard_ace:.4f}")
    print(f"Class-Adaptive Calibration - ECE: {adaptive_ece:.4f}, ACE: {adaptive_ace:.4f}")
    print(f"ECE Improvement: {standard_ece - adaptive_ece:.4f}")
    print(f"ACE Improvement: {standard_ace - adaptive_ace:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join('results', f'results_{timestamp}.txt')
    
    with open(result_file, 'w') as f:
        f.write("Class-Adaptive Network Calibration Results\n")
        f.write("=========================================\n\n")
        f.write(f"Standard Temperature Scaling - ECE: {standard_ece:.4f}, ACE: {standard_ace:.4f}\n")
        f.write(f"Class-Adaptive Calibration - ECE: {adaptive_ece:.4f}, ACE: {adaptive_ace:.4f}\n")
        f.write(f"ECE Improvement: {standard_ece - adaptive_ece:.4f}\n")
        f.write(f"ACE Improvement: {standard_ace - adaptive_ace:.4f}\n")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Reliability diagrams
    plot_reliability_diagram(
        standard_results['confidences'], 
        standard_results['predictions'], 
        standard_results['labels'],
        title="Standard Temperature Scaling",
        save_path=os.path.join('results', f'reliability_standard_{timestamp}.png')
    )
    
    plot_reliability_diagram(
        adaptive_results['confidences'], 
        adaptive_results['predictions'], 
        adaptive_results['labels'],
        title="Class-Adaptive Calibration",
        save_path=os.path.join('results', f'reliability_adaptive_{timestamp}.png')
    )
    
    # Temperature distribution
    plot_temperature_distribution(
        adaptive_calibration,
        class_names=class_names,
        save_path=os.path.join('results', f'temperature_distribution_{timestamp}.png')
    )
    
    # ECE per class
    std_ece_per_class = plot_ece_per_class(
        standard_results['confidences'],
        standard_results['predictions'],
        standard_results['labels'],
        standard_results['labels'],  # Using labels as class indices
        save_path=os.path.join('results', f'ece_per_class_standard_{timestamp}.png')
    )
    
    adp_ece_per_class = plot_ece_per_class(
        adaptive_results['confidences'],
        adaptive_results['predictions'],
        adaptive_results['labels'],
        adaptive_results['labels'],  # Using labels as class indices
        save_path=os.path.join('results', f'ece_per_class_adaptive_{timestamp}.png')
    )
    
    # Plot ECE improvement per class
    if std_ece_per_class and adp_ece_per_class:
        import matplotlib.pyplot as plt
        
        ece_improvement = np.array(std_ece_per_class) - np.array(adp_ece_per_class)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(ece_improvement)), ece_improvement)
        plt.xlabel('Class Index')
        plt.ylabel('ECE Improvement')
        plt.title('ECE Improvement per Class (Standard vs Adaptive)')
        plt.axhline(y=0, color='r', linestyle='-')  # Add line at y=0
        plt.tight_layout()
        plt.savefig(os.path.join('results', f'ece_improvement_{timestamp}.png'))
        plt.close()
        
        # Calculate and report overall statistics
        improvement_percent = len(ece_improvement[ece_improvement > 0]) / len(ece_improvement) * 100
        print(f"\nClass-wise improvement statistics:")
        print(f"  - Classes improved: {len(ece_improvement[ece_improvement > 0])}/{len(ece_improvement)} ({improvement_percent:.1f}%)")
        print(f"  - Average improvement: {np.mean(ece_improvement):.4f}")
        print(f"  - Maximum improvement: {np.max(ece_improvement):.4f}")
        print(f"  - Minimum improvement: {np.min(ece_improvement):.4f}")
    
    print(f"\nExperiment completed. Results saved to {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Class-Adaptive Network Calibration")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for calibration training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for calibration training')
    parser.add_argument('--l2_reg', type=float, default=0.01, help='L2 regularization strength')
    parser.add_argument('--train_base', action='store_true', help='Train the base model from scratch')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA training')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)