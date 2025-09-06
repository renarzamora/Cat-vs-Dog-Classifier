import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CatDogCNN

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]
)

train_loader = DataLoader(
    datasets.ImageFolder('./data/train', transform=transform),
    batch_size=32, shuffle=True
)

val_loader = DataLoader(
    datasets.ImageFolder('./data/val', transform=transform),
    batch_size=32, shuffle=True
)

model = CatDogCNN()
criterion = nn.BCEWithLogitsLoss()
optimizer  = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        labels = labels.float().unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

# Validation
model.eval()
correct, total = 0,0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        pred = torch.sigmoid(outputs)
        predicted = (pred > 0.5).int().squeeze()
        correct += (predicted == labels.int()).sum().item()
        total += labels.size(0)
    print(f"Accuracy: {100 * correct / total:.2f}%")

# save model / make sure that the folder model exists
torch.save(model.state_dict(), './model/cat_dog_cnn.pth')


