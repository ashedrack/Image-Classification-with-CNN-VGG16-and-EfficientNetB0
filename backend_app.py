from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import logging
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Path to your model file
MODEL_PATH = "fine_tuned_vgg16_model.h5"
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        logging.info("Model loaded and compiled successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # Read and process the uploaded image
        image = Image.open(io.BytesIO(await file.read()))
        image = image.convert('RGB')  # Ensure image is in RGB mode
        image = image.resize((224, 224))  # Resize to model input size
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.max(prediction)
        confidence = predicted_class  # Assuming predicted_class here represents confidence

        # Map the prediction to Cat or Dog
        if predicted_class >= 0.99:
            predicted_class_label = 1  # Cat
        else:
            predicted_class_label = 0  # Dog

        # Return result
        return {"predicted_class": int(predicted_class_label), "confidence": float(confidence)}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Could not get a prediction from the model.")

# To run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
