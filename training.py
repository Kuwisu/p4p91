
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score


class ModelTraining:
    def __init__(self,
                 model,
                 output_dir: str = "model-out",
                 output_name: str = "p4p91-emotion-model.pth",
                 train_path: str = "processed-data/test",
                 val_path: str = "processed-data/val",
                 learn_rate: float = 0.001,
                 num_epochs: int = 10,
                 mean: np.array = None,
                 std: np.array = None,
                 size: tuple[int, int] = (224, 224),
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 weight_decay: float = 1e-5
                 ):
        if mean is None:
            mean = np.array([0.5, 0.5, 0.5])
        if std is None:
            std = np.array([0.5, 0.5, 0.5])

        self.transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,)
        ])

        train_dataset = torchvision.datasets.ImageFolder(train_path, transform=self.transforms)
        self.train_size = len(train_dataset)

        val_dataset = torchvision.datasets.ImageFolder(val_path, transform=self.transforms)
        self.val_size = len(val_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learn_rate, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            train_acc = 0.0
            train_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().item() * images.size(0)
                _, prediction = torch.max(outputs, 1)

                train_acc += int(torch.sum(prediction == labels))

            train_acc = train_acc / self.train_size
            train_loss = train_loss / self.train_size

            print("Epoch " + str(epoch) + " Loss: " + str(train_loss) + " Accuracy: " + str(train_acc))
            self.evaluate_model()

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, output_name)
        torch.save(self.model.state_dict(), save_path)

    # Define Evaluation Function
    def evaluate_model(self):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # Print evaluation metrics
        print("\nModel Evaluation:")
        print(f"Standard Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
        print("Classification Report:\n", classification_report(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
