# Romulus Vlad Lapusan (s5327051)

## ML Programming Project

1. Main Project Idea: Object Recognition

- Overview:

Pneumonia detection is a critical application in medical imaging that involves the classification of chest X-rays to determine the presence of pneumonia. This project aims to develop a Convolutional Neural Network (CNN) model capable of accurately detecting pneumonia in unseen X-ray images.

- Objective:

  * Develop a CNN-based classification model for detecting pneumonia in chest X-ray images.

  * Achieve high accuracy and robustness in classifying unseen data.

  * Provide detailed insights into the model’s performance metrics, including accuracy, precision, recall, and F1-score.

  * Ensure the project’s replicability by making the code open-source and providing in-depth documentation.
    
- Prior Knowledge / Papers / Examples:

    * Liu, Li, et al. “Deep Learning for Generic Object Detection: A Survey.” International Journal of Computer Vision, vol. 128, no. 2, Feb. 2020, pp. 261–318. Springer Link, https://doi.org/10.1007/s11263-019-01247-4

    * Zaidi, Syed Sahil Abbas, et al. A Survey of Modern Deep Learning Based Object Detection Models. arXiv, 12 May 2021, arXiv.org, https://doi.org/10.48550/arXiv.2104.11892

- Repositories and Projects:

    * PyTorch: Flexible and dynamic deep learning framework.

    * Matplotlib: Visualisation library for creating plots and charts.

    * Scikit-Learn: Machine learning library for evaluating models.

2. Approach:

    - Dataset Preparation:

      * Download the Chest X-Ray dataset from Kaggle.

      * Perform data pre-processing, including resizing, normalisation, and augmentation.

      * Split the dataset into training, validation, and test sets, unless the author has already done so.

    - Model Architecture:

      * Design a CNN model tailored for medical image classification.

      * Integrate architectural improvements such as batch normalisation, dropout, and residual connections.

      * Experiment with pre-trained models like ResNet18 for feature extraction.

    - Training:

      * Train the model using the Adam optimiser and appropriate learning rate scheduling.

      * Monitor performance using metrics like loss and accuracy.

    - Evaluation:

      * Evaluate the model’s performance on the test set using accuracy, precision, recall, and F1-score.

      * Analyse the confusion matrix to identify areas for improvement.

    - Deployment:

      * Package the model and provide scripts for inference on unseen data.

4. Dataset:

The dataset used for this project is the Chest X-Ray Images (Pneumonia) dataset available on Kaggle. It contains:

   * X-ray images categorised into two classes: Normal and Pneumonia.

   * Clear labels for training and testing.

   * High-quality images suitable for training deep learning models.

Dataset Link:

   * [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

   
