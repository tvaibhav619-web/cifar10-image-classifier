import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Import your model from src
import sys
sys.path.append("../src")
from model import SimpleCNN

# CIFAR-10 class names
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Load model function
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN()
    model.load_state_dict(torch.load("../models/cifar10_cnn.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Streamlit App UI
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=200)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Prediction
    outputs = model(img_tensor)
    _, predicted = outputs.max(1)
    label = classes[predicted.item()]

    st.subheader(f"Predicted Class: **{label}**")

    # Softmax probabilities
    probs = nn.Softmax(dim=1)(outputs)
    prob_value = probs[0][predicted.item()].item() * 100
    st.write(f"Confidence: **{prob_value:.2f}%**")

    
