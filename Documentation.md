# Pneumonia Detection Models - Technical Documentation

This document provides detailed information about the model architectures, training processes, and performance metrics for all model iterations in this project.

## Model Evolution

### Primary Model

**Architecture:**
- Simple fully connected neural network with one hidden layer
- Input images resized to 56x56 pixels
- Training epochs: 10

**Performance Metrics:**
- Training Accuracy: 93.46%
- Validation Accuracy: 93.68%
- Average Test Batch Accuracy: 43.75%
- F1-Score: 0.8672

**Execution Environment:**
- Google Colab

**Key Observations:**
- Significant gap between validation (93.68%) and test accuracy (43.75%)
- Visual inspection of predictions shows poor performance on test data
- Model architecture insufficient for medical image analysis complexity
- High sensitivity for pneumonia classification, lower precision for normal classification
- Bias towards the majority class affecting performance

### Secondary Model (CNN Integration)

**Architecture:**
- CNN with 5 convolutional layers and 3 fully connected layers
- Input images resized to 224x224 pixels
- Training epochs: 20
- Added batch normalisation and dropout layers

**Performance Metrics:**
- Training Accuracy: 95.64%
- Best Validation Accuracy: 97.51%
- Average Test Batch Accuracy: 73.56%
- Total Correct Predictions: 459/624

**Execution Environment:**
- Google Colab

**Improvements:**
- Increased validation set size
- Added convolutional layers
- Implemented batch normalisation
- Integrated dropout layers

**Key Observations:**
- Test accuracy increased to 73.56%
- Most incorrect predictions are normal X-rays misclassified as pneumonia
- Continued bias towards the majority class
- Potential labelling errors in the dataset

### Third/Fourth Model (Transfer Learning)

**Architecture:**
- ResNet34 with pre-trained weights
- Custom layers added on top of base model
- Input images: 224x224 pixels
- Class weighting to address imbalance

**Implementation Features:**
- Running locally on MPS (Metal Performance Shaders)
- KFold cross-validation
- WeightedRandomSampler for class balance
- Early stopping based on validation loss
- Model saving (.pth format)

**Performance with Class Weights [1.0, 2.0]:**
- Training Accuracy: 94.53% / 98.03%
- Best Validation Accuracy: 96.36% / 97.89%
- Test Accuracy: 73.56% / 88.78%
- Total Correct: 459/624 → 554/624
- Precision: 0.85, Recall: 1.00, F1-Score: 0.92

**Performance with Class Weights [1.0, 3.7]:**
- Training Accuracy: 96.45%
- Best Validation Accuracy: 97.80%
- Test Accuracy: 87.02%
- Total Correct: 543/624
- Precision: 0.83, Recall: 0.99, F1-Score: 0.91

## Generalisation Testing

### COVID-19 Dataset Performance

**Results with Class Weights [1.0, 2.0]:**
- Test Accuracy: 21%
- Class: NORMAL
  - Precision: 0.21, Recall: 1.00, F1-Score: 0.35, Support: 40.0
- Class: COVID
  - Precision: 0.00, Recall: 0.00, F1-Score: 0.00, Support: 148.0

**Results with Class Weights [1.0, 3.7]:**
- Test Accuracy: 70%
- Class: NORMAL
  - Precision: 0.17, Recall: 0.10, F1-Score: 0.12, Support: 40.0
- Class: COVID
  - Precision: 0.78, Recall: 0.86, F1-Score: 0.82, Support: 148.0

## Technical Implementation Details

### Data Processing
- Data augmentation (random flips, rotations, colour jitter)
- Normalisation using dataset mean and standard deviation
- Weighted random sampling for class balance

### Training Optimisations
- Early stopping to prevent overfitting
- K-fold cross-validation for reliable performance estimates
- Class weighting to address imbalance

### Execution Environments
- Initial models: Google Colab
- Final models: Local execution with MPS acceleration

## Conclusion

The final model achieved 87.02% test accuracy and an F1-score of 0.91 on the pneumonia dataset, showing high sensitivity (recall: 0.99) in detecting pneumonia. This indicates effective learning of pneumonia-specific features.

When tested on a COVID-19 dataset without additional training, the model achieved 70% accuracy with an F1-score of 0.82 for COVID-19 cases, suggesting shared features between pneumonia and COVID-19 X-rays. However, it struggled with normal cases (F1-score: 0.12), showing its dependence on pneumonia patterns.

These results highlight the model's potential for generalisation while emphasising the importance of fine-tuning with domain-specific data for specialised tasks.

## Datasets
- Pneumonia Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)
- COVID-19 Dataset: [COVID-19 X-ray Dataset](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets)