
🖼️ CIFAR-10 Image Classifier (PyTorch + Streamlit)

A complete deep-learning project that trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset, saves the trained model, and provides a Streamlit web app for making predictions on images.

🚀 Built with PyTorch, Torchvision, Streamlit, and Python 3.10+.

📌 Features

✔ Train a CNN on the CIFAR-10 dataset

✔ Save the trained model (.pth file)

✔ Upload any image and get prediction

✔ Interactive Streamlit web app

✔ Simple project structure (easy to understand)

✔ GPU support (CUDA compatible)

📂 Project Structure
cifar10_project/
│── models/
│   └── cifar10_model.pth        # trained model
│── src/
│   ├── train.py                 # training script
│   └── model.py                 # CNN model definition (optional)
│── app.py                       # Streamlit web app
│── requirements.txt             # dependencies
│── README.md                    # project documentation

🧠 Dataset — CIFAR-10

CIFAR-10 contains 60,000 color images (32×32 pixels) across 10 classes:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

🚀 How to Run the Project
1️⃣ Clone this repository
git clone https://github.com/YOUR-USERNAME/cifar10-image-classifier.git
cd cifar10-image-classifier

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model

Run the training script to create the model file:

python src/train.py


This will generate:

models/cifar10_model.pth

4️⃣ Run the Streamlit Web App
streamlit run app.py


A web app will open in your browser.

🖥️ Streamlit App Preview

The web app allows you to:

Upload an image

Press Predict

See the predicted class + confidence

🧩 Technologies Used

Python

PyTorch

Torchvision

Streamlit

Pillow

NumPy

📦 requirements.txt

Your requirements.txt should include:

torch
torchvision
streamlit
Pillow
numpy
