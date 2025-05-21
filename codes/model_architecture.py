import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
import load_data

num_classes = 2
batch_size = 32
learning_rate = 0.0001
num_epochs = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = load_data.get_dataloaders(
    root_dir="C:/Users/Gayatri/Documents/task/data",
    batch_size=batch_size
)

model = models.convnext_small(pretrained=False)  
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)  

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        train_accuracy = 100 * train_correct / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
        import numpy as np
        model.eval()
        test_loss = 0
        test_correct = 0
        total_test_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                total_test_samples += labels.size(0)

        test_accuracy = 100 * test_correct / total_test_samples
        print(f"Testing Loss: {test_loss / len(test_loader):.4f}, Testing Accuracy: {test_accuracy:.2f}%")
        scheduler.step()

    return model

trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs)

torch.save(trained_model.state_dict(), "convnext_small_custom.pth")
print("Model training complete and saved!")


