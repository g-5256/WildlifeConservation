import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Function to classify the image using ResNet-50
def classify_image(image):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image)

    _, predicted_class = output.max(1)
    return predicted_class.item()

# Function to fetch animal information from Wikipedia
def get_animal_info(animal_name):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{animal_name}"
    response = requests.get(url)
    data = response.json()
    return data.get("extract", "Information not found.")

# Streamlit UI
st.title("Animal Species Classifier and Information")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the image
    class_id = classify_image(image)
    
    # Map class ID to an animal name (you may need to customize this based on your dataset)
    class_to_animal = {
        0: "cat",
        1: "dog",
        # Add more class mappings here...
    }
    
    animal_name = class_to_animal.get(class_id, "Unknown")
    
    st.subheader("Animal Species:")
    st.write(animal_name)

    # Get information about the animal from Wikipedia
    animal_info = get_animal_info(animal_name)
    
    st.subheader("Animal Information:")
    st.write(animal_info)
