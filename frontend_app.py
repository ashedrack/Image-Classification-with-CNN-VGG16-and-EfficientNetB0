import streamlit as st
import requests
from PIL import Image
import io
import tensorflow as tf
import numpy as np

# Set the FastAPI endpoint
FASTAPI_URL = "https://image-classification-with-cnn-vgg16-and.onrender.com/predict/"

# Title of the app
st.title("Image Classification with CNN | VGG16 and image validation with EfficientNetB0 ")

# Warning for users
st.warning("**Important:** Please ensure the image is correctly labeled with 'cat' or 'dog' in the filename before uploading. "
           "Images that do not have these labels will not be accepted for processing.")

# Instructions
st.write("Upload an image of a cat or dog, and the model will predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Simple model to verify image content using EfficientNetB0
def is_cat_or_dog(image):
    # Load the EfficientNetB0 model pre-trained on ImageNet
    model = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=True)

    # Preprocess the image to fit the model input requirements
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(predictions, top=5)[0]

    # List of keywords associated with cats and dogs
    cat_keywords = ["cat", "kitten"]
    dog_keywords = ["dog", "puppy"]

    # Check if any of the top predictions match cat or dog keywords
    for _, label, _ in decoded_predictions:
        if any(keyword in label.lower() for keyword in cat_keywords + dog_keywords):
            return True
    return False

if uploaded_file is not None:
    # Get the file name
    file_name = uploaded_file.name.lower()

    # Check if the file name contains "cat", "ca", "dog", or "do"
    if not any(keyword in file_name for keyword in ["cat", "ca", "dog", "do"]):
        st.error("Sorry, this image must be labeled with the correct name for it to be processed. "
                 "Please ensure the file name includes one of the following: 'cat', 'ca', 'dog', or 'do'.")
    else:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Validate the content of the image
        content_valid = is_cat_or_dog(image)

        # Allow dog images to be processed even if content validation fails, as long as the filename suggests it's a dog
        if not content_valid and "dog" not in file_name and "do" not in file_name:
            st.error("This kind of image was not trained to be processed for the prediction. "
                     "Please upload an image containing a cat or a dog.")
        else:
            # Button to make prediction
            if st.button("Predict"):
                # Convert the image to bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                # Send the image to the FastAPI endpoint
                response = requests.post(
                    FASTAPI_URL,
                    files={"file": (file_name, img_bytes, "image/png")}
                )

                if response.status_code == 200:
                    result = response.json()
                    if "predicted_class" in result:
                        # Map the predicted class to either "Cat" or "Dog"
                        predicted_class = "Cat" if result['predicted_class'] == 1 else "Dog"
                        st.write(f"**Predicted Class:** {predicted_class}")
                        st.write(f"**Confidence:** {result['confidence']:.2f}")
                    elif "error" in result:
                        st.write(result["error"])
                else:
                    st.write("Error: Could not get a prediction from the model.")

# To run the Streamlit app use: streamlit run app.py
