import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# with open("tree_data_labels.txt") as f:
#     labels = [line.strip() for line in f.readlines()]

dir = "Images"

tree_names = []

class TreeDataset(Dataset):
    def __init__(self, files: dict, transform=None):
        self.files = files
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self.files[idx].get("data")
        label = self.files[idx].get("label")
        if self.transform:
            img = self.transform(img)
        return img, label

with open("treeNames.txt", "r") as f:
    for line in f:
        tree_names.append(line.strip())


def get_label_from_filename(filename):
    for i, tree_name in enumerate(tree_names):
        if tree_name in filename:
            return i
    return None

processed_imgs = []
    
for filename in os.listdir(dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        true_label = get_label_from_filename(filename)
        if true_label is not None:
          path = os.path.join(dir, filename)
          image = Image.open(path)
          processed_imgs.append({"label": true_label,
                                 "data": image})

size = len(tree_names)
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, size)  # Adjust the final layer to match the number of classes
model.train()

lf = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

processed_imgs = random.sample(processed_imgs, len(processed_imgs))
train, test = train_test_split(processed_imgs, test_size=0.3)

train_dataset = TreeDataset(train, transform=preprocess)
test_dataset = TreeDataset(test, transform=preprocess)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

epochs = 500

print("Training...")
for epoch in range(epochs):
    tot_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lf(outputs, labels)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {tot_loss}")
print("Testing...")

identified = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        identified += (predicted == labels).sum().item()
        predicted_trees = [tree_names[pred] for pred in predicted]
        true_trees = [tree_names[label] for label in labels]
        print("Predicted: ", predicted_trees)
        print("True: ", true_trees)
        print("Accuracy: ", identified / total)
        print()