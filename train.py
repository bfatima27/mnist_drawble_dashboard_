from model import Net
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

# -----------------------------
# EarlyStopping class
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# -----------------------------
# Dataset
# -----------------------------
transform = transforms.ToTensor()
train_dataset = datasets.MNIST("dataset", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

# -----------------------------
# Device və Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# -----------------------------
# Loss və Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
early_stopper = EarlyStopping(patience=5, verbose=True)

# -----------------------------
# Training loop
# -----------------------------
epochs = 50  # böyük epoch sayı qoymaq olar, early stopping dayandıracaq
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss:.4f}")

    # Check early stopping
    if early_stopper(epoch_loss, model):
        print(f"Early stopping at epoch {epoch+1}")
        break

# -----------------------------
# Save the best model
# -----------------------------
torch.save(early_stopper.best_model, "cnn_mnist.pth")
print("Model saved!")