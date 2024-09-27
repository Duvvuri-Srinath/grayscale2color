import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image 

# Function to download the model if not already present
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        st.write(f"Downloading model from {model_url}...")
        urllib.request.urlretrieve(model_url, model_path)
        st.write("Model downloaded successfully!")

# Paths to download the model
DIR = r"./modal"
os.makedirs(DIR, exist_ok=True)
PROTOTXT = os.path.join(DIR, r"colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"pts_in_hull.npy")
MODEL = os.path.join(DIR, r"colorization_release_v2.caffemodel")

# URL for the model
MODEL_URL = "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1"

# Download the model if not already available
download_model(MODEL_URL, MODEL)

# Load the model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Function to process the image and colorize it
def colorize_image(image):
    # Convert the uploaded file to OpenCV format
    image = np.array(image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Convert back to 8-bit image
    colorized = (255 * colorized).astype("uint8")
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colorized)

# Streamlit UI
def main():
    st.title("Image Colorization")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload a black and white image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Open and display the uploaded image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption='Uploaded Image', use_column_width=True)

        # Process and display the colorized image
        colorized_image = colorize_image(input_image)
        st.image(colorized_image, caption='Colorized Image', use_column_width=True)

if __name__ == '__main__':
    main()
