# -*- coding: utf-8 -*-

'''
This script evaluates the trained model’s accuracy on the test dataset and compares it to the previously recorded best validation accuracy after a 20 epochs run.
It offers an overview of how well the model generalises to new, unseen data, providing insights into model's peformance and potential areas for improvement.
'''

import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class ModelTester:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.image_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def test_model(self, test_folder, batch_size=16):
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"Test folder {test_folder} does not exist.")

        test_data = datasets.ImageFolder(test_folder, transform=self.image_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

        total_correct = 0
        total_images = 0
        batch_accuracies = []

        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                correct = (predictions == labels).sum().item()
                batch_accuracy = (correct / labels.size(0)) * 100
                batch_accuracies.append(batch_accuracy)
                total_correct += correct
                total_images += labels.size(0)

        final_accuracy = (total_correct / total_images) * 100
        avg_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)

        print(f"Results Summary: Overall Accuracy: {final_accuracy:.2f}% Average Batch Accuracy: {avg_batch_accuracy:.2f}% Total Correct: {total_correct}/{total_images}")
        return final_accuracy

    def show_example_predictions(self, test_folder, num_images=16):
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"Test folder {test_folder} does not exist.")

        test_data = datasets.ImageFolder(test_folder, transform=self.image_transform)
        loader = DataLoader(test_data, batch_size=num_images, shuffle=True)
        images, labels = next(iter(loader))
        images = images.to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]

        plt.figure(figsize=(20, 20))
        for i in range(num_images):
            plt.subplot(4, 4, i + 1)
            img = images[i].cpu().squeeze()
            plt.imshow(img, cmap='gray')
            true_label = test_data.classes[labels[i]]
            pred_label = test_data.classes[predictions[i]]
            conf = confidence[i].item()
            color = 'green' if predictions[i] == labels[i] else 'red'
            plt.title(f'Truth: {true_label} Guess: {pred_label} Confidence: {conf:.2f}', color=color, fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

def test_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tester = ModelTester(model, device)
    test_folder = '/content/chest_xray/chest_xray/test'

    print("Model Testing")
    print(f"Using device: {device}")

    try:
        accuracy = tester.test_model(test_folder)
        tester.show_example_predictions(test_folder)
        print("Performance Comparison:")
        print("Best Validation Accuracy: 97.89%")
        print(f"Test Accuracy: {accuracy:.2f}%")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

test_model(model)