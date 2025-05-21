
import os
from collections import Counter

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class ModelTraining:
    def __init__(self,
                 model,
                 output_dir: str = "model-out",
                 log_name: str = "training-log.txt",
                 model_name: str = "p4p91-emotion-model.pth",
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

        # Establish an output directory and create a file to record console output
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        log_name = os.path.join(self.output_dir, log_name)
        open(log_name, "x").close()

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

        self.text_labels = train_dataset.classes
        self.numeric_labels = np.arange(len(self.text_labels))
        label_counts = Counter(train_dataset.targets)
        class_counts = np.array([label_counts[i] for i in range(len(train_dataset.classes))])
        class_weights = 1.0 / class_counts
        class_weights = class_weights / np.sum(class_weights)

        self.train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learn_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(self.device))

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

            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']

            train_acc = train_acc / self.train_size
            train_loss = train_loss / self.train_size

            train_log = f"Epoch {epoch+1} Loss: {train_loss:.4f} Accuracy: {train_acc:.4f} Learning Rate: {prev_lr:.6f} => {new_lr:.6f}"
            eval_log = self.evaluate_model(epoch+1)

            print(train_log)
            print(eval_log)

            file = open(log_name, "a")
            file.write(train_log)
            file.write("\n")
            file.write(eval_log)
            file.write("\n")
            file.close()

        save_path = os.path.join(output_dir, model_name)
        torch.save(self.model.state_dict(), save_path)

    # Define Evaluation Function
    def evaluate_model(self, epoch):
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

        # Prepare and save confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.numeric_labels)
        ConfusionMatrixDisplay(cm, display_labels=self.text_labels).plot(cmap='Blues')
        plt.title(f"Epoch {epoch} Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, f"confusion-matrix-{epoch}.png"))
        plt.close()

        # Return an evaluation log
        eval_log = f"Classification Report: \n{classification_report(
            y_true, y_pred, target_names=self.text_labels, zero_division=0)}"
        return eval_log
