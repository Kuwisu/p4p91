
import os

import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision

class ModelTraining:
    def __init__(self,
                 model,
                 train_path: str = "processed-data/test",
                 val_path: str = "processed-data/val",
                 learn_rate: float = 0.001,
                 num_epochs: int = 10,
                 mean: np.array = None,
                 std: np.array = None,
                 size: tuple[int, int] = (224, 224),
                 batch_size: tuple[int, int, ...] = (32, 32),
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

        train_loader = DataLoader(
            torchvision.datasets.ImageFolder(train_path, transform=self.transforms),
            batch_size=batch_size[0], shuffle=True
        )
        val_loader = DataLoader(
            torchvision.datasets.ImageFolder(val_path, transform=self.transforms),
            batch_size=batch_size[1], shuffle=True
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=learn_rate, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        train_count = len(glob.glob(os.path.join(train_path, "*")))
        val_count = len(glob.glob(os.path.join(val_path, "*")))

        for epoch in range(num_epochs):
            self.model.train()
            train_acc = 0.0
            train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                train_acc += int(torch.sum(prediction == labels.data))

            train_acc = train_acc / train_count
            train_loss = train_loss / train_count

            print("Epoch " + str(epoch) + " Train loss: " + str(train_loss) + " Train Accuracy: " + str(train_acc))