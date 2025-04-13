# Class-Adaptive Network Calibration

This project is a reimplementation of the paper "Class-Adaptive Network Calibration" (Liu et al., CVPR 2023). It explores class-specific temperature scaling for improving neural network calibration.

## Project Structure

- `src/`: Source code
  - `models.py`: Calibration model implementations
  - `training.py`: Training and evaluation functions
  - `metrics.py`: Calibration metrics (ECE, ACE)
  - `visualization.py`: Visualization utilities
- `main.py`: Main script to run experiments
- `demo.py`: Script for generating demo visualizations
- `requirements.txt`: Dependencies

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`
4. Generate demo visualizations: `python demo.py`