import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

IMG_SIZE = 128
MODEL_PATH = 'models/deepfake_model.h5'

# Load trained model
model = load_model(MODEL_PATH)

# Load and preprocess test image
def prepare_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path '{image_path}' not found.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Run prediction
def predict(image_path):
    try:
        image = prepare_image(image_path)
        prediction = model.predict(image)[0][0]
        if prediction < 0.5:
            print("✅ Prediction: REAL FACE (Label = 0)")
        else:
            print("❌ Prediction: FAKE FACE (Label = 1)")
        print(f"🔢 Confidence score: {prediction:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# Run from command line: python predict.py path_to_image.jpg
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict(sys.argv[1])
