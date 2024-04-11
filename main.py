import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# MLP architecture
class MLP(nn.Module): # All models should inherit from nn.Module
    def __init__(self, num_input, hidden_size, num_classes):
        super(MLP, self).__init__()

        #ALLGOOD
        self.fc1 = nn.Linear(num_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)

        # WE NEED TO SWITCH THIS WITH AN APPROPRIATE ACTIVATION FUNCTION
        x = F.relu(self.fc1(x))
        # ^^^
        
        x = self.fc2(x)
        return x

# CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model initialization
mlp_model = MLP(28*28, 128, 10).to(device)
cnn_model = CNN(10).to(device)

# Define optimizer and loss function
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training MLP model
for epoch in range(5):
    train_loss = train_model(mlp_model, train_loader, optimizer_mlp, criterion, device)
    print(f'MLP Epoch: {epoch+1}, Loss: {train_loss:.4f}')

# Testing MLP model
mlp_accuracy = test_model(mlp_model, test_loader, device)
print(f'MLP Accuracy: {mlp_accuracy:.4f}')

# Training CNN model
for epoch in range(5):
    train_loss = train_model(cnn_model, train_loader, optimizer_cnn, criterion, device)
    print(f'CNN Epoch: {epoch+1}, Loss: {train_loss:.4f}')

# Testing CNN model
cnn_accuracy = test_model(cnn_model, test_loader, device)
print(f'CNN Accuracy: {cnn_accuracy:.4f}')
