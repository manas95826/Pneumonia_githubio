from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

model = load_model("keras_model.h5", compile=False)

class_names = ["0", "1"]  # Make sure the class names match the order in your model

def main():
    st.title("Image Classification")
    st.header("Pneumonia X-Ray Classification")
    st.text("Upload a Pneumonia X-Ray for classification")
    
    file = st.file_uploader('Upload an image file', type=["jpg", "jpeg"])
    if file is not None:
        img = Image.open(file)
        
        # Check if the image is in RGB format
        if img.mode != "L":
            st.warning("Please re-upload the X-ray file only")
            return

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        img = img.convert('RGB')  # Convert to RGB mode to ensure 3 channels
        size = (224, 224)
        image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)

        # Turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predict the model
        confidence_score = model.predict(data)
        predicted_class = np.argmax(confidence_score)

        if predicted_class == 1:
            st.success("Pneumonia found!")
        else:
            st.error("No Pneumonia Found.")

if __name__ == "__main__":
    main()
