import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN   # import your model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load CIFAR-10 test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False)

# Load trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("../models/cifar10_cnn.pth", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

