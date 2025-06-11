Pneumonia Detection Models

Primary:

- Simple fully connected neural network with one hidden layer
- Input images resized to 56x56 pixels
- Training epochs: 10
- Currently running on Google Colab
- Future Support: Local execution, Metal FX optimisation*

Secondary(After CNN Integration):

- Convolutional Neural Network(CNN):
                                    5 convolutional layers.
                                    3 fully connected layers.
- Input images resized to a higher resolution (224x224)
- Training epochs: 20
- Currently running on Google Colab
- Changes: 
         Increased validation set size,
         Added Convolutional Layers, 
         Implemented Batch Normalisation, 
         Integrated Dropout Layers
- Future Support: Local execution, potential Metal FX optimisation*

Third/Fourth:

- Running Locally
- CNN Architecture
- Mutliple classes for training, validation and testing
- Includes Constants, Dropout Rates, Transforms, KFold, WeightedRandomSampler etc.
- ResNet34 with pre-trained weights
- State_dict with weights_only = True for ToTensor Warning Error when loading the third model's weights
- Integrated the ResNet34 model with additional custom layers
- The pre-trained weights are downloaded and cached locally to ensure faster loading in future uses 
   Using from torchvision import models, it's possible to instantly switch between pre-trained models like ResNet18 and ResNet34 while maintaining code integrity
- Model saving after training .pth
- Metrics Calculations & Predictions Visualisation plt

Performance Metrics Primary:

- Training Accuracy: 93.46%
- Validation Accuracy: 93.68%
- Average Test Batch Accuracy: 43.75%

Performance Metrics Secondary:

- Training Accuracy: 95.64%
- Best Validation Accuracy: 97.51%
- Average Test Batch Accuracy: 73.56%
- Total Correct: 459/624

Performance Metrics Third/Fourth:

These are the result after trying to solve the class imbalance [FModel] {class_weights = torch.tensor([1.0, 2.0]).to(device)}:

(Pneumonia Data)
- Training Accuracy: 94.53% / 98.03%
- Best Validation Accuracy: 96.36% / 97.89%
- Test Accuracy: 73.56% / 88.78%
- Total Correct: 459/624
                  554/624
- Fourth Model: 
               _Precision: 0.85
               _Recall: 1.00
               _F1-Score: 0.92

(COVID19 Data)
Test Accuracy: 0.21
Class: NORMAL
  _Precision: 0.21
  _Recall: 1.00
  _F1-Score: 0.35
  _Support: 40.0
Class: COVID
  _Precision: 0.00
  _Recall: 0.00
  _F1-Score: 0.00
  _Support: 148.0

These are the result after trying to solve the class imbalance [FModel] {class_weights = torch.tensor([1.0, 3.7]).to(device)}:

(Pneumonia Data)
- Training Accuracy: 96.45%
- Best Validation Accuracy: 97.80%
- Test Accuracy: 87.02%
- Total Correct: 543/624
- Metrics: 
               _Precision: 0.83
               _Recall: 0.99
               _F1-Score: 0.91

(COVID19 Data)
Test Accuracy: 0.70
Class: NORMAL
   _Precision: 0.17
   _Recall: 0.10
   _F1-Score: 0.12
   _Support: 40.0
Class: COVID
   _Precision: 0.78
   _Recall: 0.86
   _F1-Score: 0.82
   _Support: 148.0

Key Initial Observations:

- Significant gap between validation (93.68%) and test batch accuracy (43.75%).
- Visual inspection of predictions shows poor performance on test data.
- Current model architecture is insufficient for the complexity of medical image analysis.
- Imbalanced datasets: 
                     The datasets are imbalanced, but the mean and standard deviations are good.
                     The small test dataset may lead to statistical errors.
- F1 score (0.8672) used for performance evaluation, considering class imbalance.
- Confusion matrix insights: Misclassifications highlight the model’s ability to distinguish between classes.
- Pneumonia classification: High sensitivity.
- Normal classification: Lower precision.
- Bias towards the majority class affects performance.

After Implementing the CNN Architecture:

- Validation Accuracy: Improved to 95.42%.
- Test Accuracy: Increased to 71.47%.
- Performance Improvement: Requires more data augmentation and regularisation to improve performance.
- During predictions example visualisation I noticed potentially wrong labels.
- Most of the incorrect predictions are for normal X-ray images that are being misclassified as pneumonia.
   This suggests a continous bias towards the majority class, but in practical terms, it’s not particularly unusable.
   The only caveat is that the results must be verified by medical professionals.

After Rebuilding the Model and Training it Locally:

- Using device: MPS
- Added Constants for Dropout Rates (Conv_Layers, FuCn_Layers).
- Introduced KFold cross validation.
- Data Split: No significant changes from 8/2 to 7/3
- Training epochs: 20
- The model doesn't seem to be improving much, the best result I could achieve after multiple tries was 73.56% for Average Batch Test Accuracy.
- The datasets may contain labelling errors. 

After Transfer Learning and Fine-Tuning:

- I ended up using ResNet34 as the base model with pre-trained weights, combining it with my previous model.
- Random sampling appears to have worked decently well in ensuring fair representation during training, particularly for class imbalance.
- The implemented early stopping based on validation loss used to prevent overfitting works accordingly.
- After numerous changes, the model significantly improved, scoring very high in test accuracy.
- Analysing the metrics, I found that the model has high precision and recall for pneumonia detection, suggesting strong sensitivity for signs of the illness.
- Examining the test set predictions, I could better compare them and identify biases. 
- The dataset does seem to have certain shortcomings in terms of usability, but offers a good testing ground for detection models.

After Experimenting with Another Slightly Dfferent Dataset:

The model achieved strong performance on its intended pneumonia detection task. The results that I’ve documented throughout the whole process indicate the model effectively identifies pneumonia cases with high sensitivity and precision. 
However, when tested on a COVID-19 dataset without prior training or fine-tuning, the model performed poorly at first, achieving a test accuracy of only 21%. 
Nevertheless, after making a minor modification to the pneumonia detection model’s training process to address class imbalance by adjusting the class weights, the test accuracy has shown remarkable improvement, reaching a significant milestone of 70%. 
This enhancement in accuracy is accompanied by a continuously high accuracy score on the Pneumonia dataset.

Conclusion:

The model performed well on the pneumonia dataset, with an 87.02% test accuracy and an F1-score of 0.91, and showed high sensitivity (recall: 0.99) in detecting pneumonia. This idicates the effective learing of pneumonia specific features.
When tested on a COVID-19 dataset without extra training, the model achieved a 70% test accuracy and an F1-score of 0.82 for COVID-19 cases. This indicates some shared features between pneumonia and COVID-19 X-rays, like lung abnormalities. 
However, it struggled with normal cases, achieving an F1-score of 0.12, showing its dependence on pneumonia patterns.

These results highlight the model’s potential for generalisation, but emphasise the importance of fine-tuning with domain-specific data to enhance performance on various tasks, such as COVID-19 detection.

Planned Improvements:

1. Architecture Changes:
   - Switch to CNN architecture (implemented)
   - Increase image resolution from 56x56 to 224x224 (implemented)
   - Transfer learning with pre-trained models (implemented{ResNet34})

2. Data Processing:
   - Implement data augmentation to increase data variability (implemented)
   - Add proper image normalisation (implemented)
   - Enhance preprocessing pipeline (implemented)

3. Training Optimisations:
   - Add regularisation techniques (stronger dropout, weight decay)
   - Consider early stopping (implemented)
   - Use k-fold cross validation for more reliable performance estimates (implemented)

4. Platform Support:
   - Current: Google Colab (Previous Two Iterations)
   - Planned: Local execution support (implemented)
   - Future: Potential Metal FX support (Runs on MPS)

Dataset:
Using Kaggle chest X-ray dataset for pneumonia detection:
    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
