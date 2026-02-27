# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-27

### Added
- **Initial release** of xai-lab - a config-driven sandbox for training deep learning models and producing Explainable AI (XAI) outputs
- **Core framework** with modular architecture:
  - Config-driven experiment management using YAML files
  - Modular library structure under `src/xai_lab/`
  - Separate modules for core, data, models, explainers, and utilities
- **Deep Learning Model Support**:
  - ResNet18 implementation with pretrained weights support
  - Training engine with AdamW optimizer and configurable hyperparameters
  - Class balancing support via weighted sampling and class weights
- **Dataset Management**:
  - CSV-based dataset loading with metadata support
  - Stratified train/val/test split generation
  - Image preprocessing pipeline with configurable augmentations
- **Explainable AI Methods**:
  - Saliency maps (input gradients) implementation
  - Grad-CAM implementation for better visualization
  - Explanation generation for high/low confidence predictions
- **Evaluation & Metrics**:
  - Model evaluation with confusion matrix
  - Macro-F1 score calculation
  - Test summary reporting
- **Visualization Tools**:
  - Training loss and validation accuracy plots
  - Confusion matrix visualization
  - Saliency map overlays on original images
- **CLI Scripts**:
  - `train.py` - Model training with config support
  - `evaluate.py` - Model evaluation and metrics generation
  - `explain.py` - XAI explanation generation
  - `make_ckplus_splits.py` - Dataset preprocessing
  - `make_report_plots.py` - Visualization generation
- **Configuration System**:
  - Hierarchical YAML configs for experiments, models, data, and training
  - First experiment: `exp001_ckplus_resnet18` for facial emotion classification
- **Documentation**:
  - Comprehensive README with setup instructions and quickstart guide
  - Example results and visualization outputs
- **Development Setup**:
  - Requirements management with separate PyTorch dependencies
  - Git ignore configuration for data and artifacts
  - MIT License

### Features
- **Facial Emotion Classification**: Initial focus on CK+ dataset with 7 emotion classes
- **Reproducible Experiments**: All configurations tracked in Git for full reproducibility
- **Modular Design**: Easy to extend with new models, datasets, and XAI methods
- **Artifact Management**: Organized output structure for runs, reports, and visualizations

### Technical Details
- Built with PyTorch for deep learning
- Pandas for data manipulation
- Scikit-learn for metrics and data splitting
- Matplotlib for visualizations
- PIL for image processing

### Dataset Support
- CK+ (Extended Cohn-Kanade) dataset integration
- Folder-based dataset structure support
- Configurable metadata CSV format

### Roadmap Preparation
- Foundation laid for future experiments with FER2013/FER+ datasets
- Architecture ready for additional XAI methods (SmoothGrad, integrated gradients)
- Prepared for MongoDB integration for experiment tracking
