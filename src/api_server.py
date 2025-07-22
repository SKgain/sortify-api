from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration ---
# Update this path if your model file is in a different location
MODEL_PATH = 'waste_classifier_final.keras'
IMAGE_SIZE = (224, 224) # Must be the same size the model was trained on
CLASS_NAMES = ['Compost', 'General_Waste', 'Hazardoos', 'Recycle']

# --- Load the trained model ---
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def preprocess_image(img_file):
    """ Loads and preprocesses an image file for the model. """
    img = image.load_img(io.BytesIO(img_file.read()), target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


@app.route('/predict', methods=['POST'])
def predict():
    """ Handles the image prediction request. """
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        processed_image = preprocess_image(file)
        predictions = model.predict(processed_image)
        
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]) * 100)

        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


# --- Run the Flask App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your local network
    app.run(host='0.0.0.0', port=5000)