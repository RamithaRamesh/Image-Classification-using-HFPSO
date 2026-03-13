HFPSO-Based Feature Selection for Image Classification
Overview

This project implements a hybrid feature selection framework combining ResNet50 deep feature extraction with Hybrid Firefly Particle Swarm Optimization (HFPSO) to improve image classification performance.

The pipeline:

Image preprocessing

ResNet50 feature extraction

Classical ML baseline evaluation

HFPSO feature selection

Model performance comparison

The system is designed for research experimentation and comparative evaluation of feature selection methods in deep learning pipelines.

Pipeline Architecture
Dataset
   ↓
Data Augmentation
   ↓
ResNet50 Feature Extractor
   ↓
Feature Vectors
   ↓
Baseline ML Models
   ↓
HFPSO Feature Selection
   ↓
Optimized Feature Set
   ↓
Model Retraining
   ↓
Performance Comparison
Features

Automatic dataset discovery

Supports datasets with or without predefined splits

Transfer learning with ResNet50

Fine-tuning of CNN layers

Hybrid Firefly Particle Swarm Optimization

Multiple classifiers:

SVM

Random Forest

KNN

Kernel SVM

Neural Network

Evaluation metrics:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Dataset Structure

Supported formats:

Option 1: Pre-split dataset
dataset/
   train/
      class1/
      class2/
   validation/
      class1/
      class2/
   test/
      class1/
      class2/
Option 2: Raw dataset
dataset/
   class1/
   class2/

The script automatically performs a 70/10/20 split.

Installation
pip install tensorflow scikit-learn pandas numpy matplotlib pillow
Running the Pipeline
python main.py --dataset path/to/dataset

Example:

python main.py --dataset IIITDMJ_Smoke
Output Files
File	Description
baseline_model_results.csv	Performance of classifiers before feature selection
fine_tuned_model_results.csv	Performance after HFPSO
fine_tuned_features_mask.npy	Selected feature mask
best_confusion_matrix.png	Confusion matrix of best model
HFPSO Algorithm

HFPSO combines:

Particle Swarm Optimization

velocity update

global best search

and

Firefly optimization principles

attraction-based movement

exploration capability

This hybrid approach improves the ability to escape local minima during feature selection.

Evaluation Metrics

The system reports:

Validation Accuracy

Test Accuracy

Precision

Recall

F1 Score

Training Time

Confusion Matrix

Research Applications

This framework can be applied to:

medical image classification

fire/smoke detection

object recognition

remote sensing

defect detection

Future Improvements

Potential extensions include:

Vision Transformer features

advanced feature selection methods

hyperparameter optimization

cross-validation experiments
