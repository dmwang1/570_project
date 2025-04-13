# Import main components for easy access
from .models import ClassAdaptiveCalibration, StandardTemperatureScaling, get_pretrained_model
from .training import train_calibration, evaluate_calibration, get_calibration_results
from .metrics import expected_calibration_error, class_adaptive_calibration_error
from .visualization import plot_reliability_diagram, plot_temperature_distribution, plot_ece_per_class