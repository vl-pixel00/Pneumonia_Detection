# -*- coding: utf-8 -*-

'''
This script is a continuation of the Primary_PneumoniaModel.py script.
It offers a simple PyTorch training setup for pneumonia detection using a Convolutional Neural Network (CNN) classifier.
It employs a conventional CNN architecture followed by a fully connected layer for the final classification.
The training process utilises the Adam optimiser without any additional complexity, making it easy to understand and maintain.

This code structure and architecture closely resemble many publicly available repositories and tutorials.
For instance, you can find similar CNN classification code patterns in various Kaggle notebooks and other PyTorch image classification examples, 
such as the ASLPart2CNN.ipynb notebook and many more.

References:
Available at:
- Kaggle. (2018). Chest X-Ray Images (Pneumonia). Available at: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/code [Accessed: 14 Dec. 2024].
- NCCA, (n.d.). jmacey. (2024). ASLPart2CNN.ipynb. Available at: https://github.com/NCCA/SEForMedia/blob/main/ASL/ASLPart2CNN.ipynb [Accessed: 14 December 2024]. 
- NCCA, (n.d.). jmacey. (2024). ASLPart3DataAugmentation.ipynb. Available at: https://github.com/username/repository/blob/main/ASL/ASLPart3DataAugmentation.ipynb [Accessed: 14 December 2024].
'''

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import struct
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import zipfile
import os
from google.colab import drive
import shutil
from tqdm import tqdm
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import pandas as pd
from torchvision import datasets
import torch.optim as optim
from torch.utils.data import random_split

drive.mount('/content/drive')

colab_notebooks_path = '/content/drive/MyDrive/Colab_Notebooks'

sys.path.append(colab_notebooks_path)

DATASET_LOCATION = '/content/chest_xray'
pathlib.Path(DATASET_LOCATION).mkdir(parents=True, exist_ok=True)

!mkdir -p ~/Downloads
!curl -L -o ~/Downloads/chest-xray-pneumonia.zip \
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia

!unzip -q -o ~/Downloads/chest-xray-pneumonia.zip -d {DATASET_LOCATION}

print(f"Dataset location: {DATASET_LOCATION}")


# In case there is not available path for custom utils
# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')



# Check dataset existence and structure after download
DATASET_LOCATION = '/content/chest_xray/chest_xray'

print(f"Dataset path exists: {os.path.exists(DATASET_LOCATION)}")
if os.path.exists(DATASET_LOCATION):
    print("Directory contents:")
    print(os.listdir(DATASET_LOCATION))

    # Dataset Structure
print("Checking dataset structure...")
for split in ['train', 'test', 'val']:
    for category in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(DATASET_LOCATION, split, category)
        if os.path.exists(path):
            num_images = len(os.listdir(path))
            print(f"{split}/{category}: {num_images} images")
        else:
            print(f"Warning: Path {path} does not exist!")


import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 1
BATCH_SIZE = 32

class PneumoniaDetectionModel(nn.Module):
    def __init__(self):
        super(PneumoniaDetectionModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),

            nn.Linear(256, 2)
        )

        self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model = PneumoniaDetectionModel()
print(model)


class PneumoniaTrainer:
    def __init__(self, model, data_dir, device=None, batch_size=32):
        self.data_dir = data_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop((IMAGE_WIDTH, IMAGE_HEIGHT), scale=(.9, 1), ratio=(1, 1)),
            transforms.ColorJitter(brightness=.2, contrast=.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.prepare_data()

    def prepare_data(self):
        full_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'chest_xray', 'train'), transform=self.train_transform)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = self.val_transform

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    def train(self, epochs=20, learning_rate=0.001):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_val_acc = 0
        best_model_state = None
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                train_pbar.set_postfix({
                    'loss': running_loss/(train_pbar.n + 1),
                    'acc': 100.*correct/total
                })

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100. * correct / total

            # Validation phase
            val_loss, val_acc = self.validate(loss_function)

            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {best_val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping triggered. Best Val Acc: {best_val_acc:.2f}%')
                break

            scheduler.step()

    def validate(self, loss_function):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    
    # Start training
DATASET_LOCATION = '/content/chest_xray'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PneumoniaDetectionModel()
trainer = PneumoniaTrainer(model, DATASET_LOCATION, device=device)
best_val_acc = trainer.train(epochs=20, learning_rate=0.001)