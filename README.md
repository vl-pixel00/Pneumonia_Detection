# Pneumonia Detection from X-Ray Images

## Overview

Pneumonia detection is a critical application in medical imaging that involves the classification of chest X-rays to determine the presence of pneumonia. This project develops a series of increasingly sophisticated deep learning models to accurately detect pneumonia in chest X-ray images, culminating in a ResNet34-based model that achieves 87.02% test accuracy.

## Project Objectives

- Develop CNN-based classification models for detecting pneumonia in chest X-ray images
- Achieve high accuracy and robustness in classifying unseen data
- Provide detailed insights into model performance metrics
- Ensure replicability through open-source code and comprehensive documentation
- Test model generalisation on COVID-19 X-ray data

## Models Implemented

1. **Primary Model**: 
   - Simple fully connected neural network with one hidden layer
   - Input images: 56×56 pixels
   - Training accuracy: 93.46%, Test accuracy: 43.75%

2. **Secondary Model**:
   - CNN with 5 convolutional layers and 3 fully connected layers
   - Higher resolution images (224×224 pixels)
   - Added batch normalisation and dropout for regularisation
   - Training accuracy: 95.64%, Test accuracy: 73.56%

3. **Advanced Models (Third/Fourth)**:
   - Transfer learning with pre-trained ResNet34
   - Fine-tuned with custom layers for pneumonia detection
   - Implemented class balancing via weighted sampling
   - Training accuracy: 96.45%, Test accuracy: 87.02%
   - F1-Score: 0.91

## COVID-19 Generalisation Test

The final model was tested on a COVID-19 X-ray dataset without additional training:
- Test accuracy: 70%
- F1-score for COVID detection: 0.82
- F1-score for normal cases: 0.12

## Key Features

- **Data Augmentation**: Enhancing model generalisation
- **Class Balancing**: Weighted sampling to address class imbalance
- **Transfer Learning**: Leveraging pre-trained ResNet34 weights
- **Cross-Validation**: K-fold cross-validation for reliable performance estimation
- **Comprehensive Metrics**: Precision, recall, F1-score, and confusion matrices

## Dataset

The project uses the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) from Kaggle, which contains:
- X-ray images categorised into two classes: Normal and Pneumonia
- Clear labels for training and testing
- High-quality images suitable for deep learning

For generalisation testing, the project uses the [COVID-19 X-ray Dataset](https://www.kaggle.com/datasets/khoongweihao/covid19-xray-dataset-train-test-sets).

## Academic References

- Liu, Li, et al. "Deep Learning for Generic Object Detection: A Survey." International Journal of Computer Vision, vol. 128, no. 2, Feb. 2020, pp. 261–318. [Springer Link](https://doi.org/10.1007/s11263-019-01247-4)
- Zaidi, Syed Sahil Abbas, et al. "A Survey of Modern Deep Learning Based Object Detection Models." arXiv, 12 May 2021. [arXiv.org](https://doi.org/10.48550/arXiv.2104.11892)

## Tools & Technologies

- **PyTorch**: Deep learning framework
- **Matplotlib**: Visualisation library
- **Scikit-Learn**: Machine learning evaluation metrics
- **Apple MPS**: Metal Performance Shaders for local training acceleration

## Project Structure

The project contains multiple Jupyter notebooks demonstrating the evolution of the models:
- `Primary_PneumoniaModel.ipynb`: Basic fully connected model
- `Secondary_PneumoniaModel_Train&Test.r0.ipynb`: CNN implementation
- `Fourth_PnemumoniaModel_T+RN34.ipynb`: Transfer learning with ResNet34
- `CombinedModels_Test.ipynb`: Testing on COVID-19 dataset

## Ethics

This project is for educational purposes only and not intended for clinical use. See [ETHICS.md](ETHICS.md) for full ethical considerations.

## Documentation

For detailed technical documentation, model performance metrics, and implementation details, see [DOCUMENTATION.md](DOCUMENTATION.md).