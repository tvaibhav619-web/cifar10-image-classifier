import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def predict(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN()
    model.load_state_dict(torch.load("../models/cifar10_cnn.pth", map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)

    outputs = model(img)
    _, predicted = outputs.max(1)
    print("Predicted:", classes[predicted.item()])

predict(r"C:\Users\tvaib\Downloads\Golden-Retriever.webp")
