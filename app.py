import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

# Initialize Flask App
app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'model.h5'  # Path to your trained .h5 file
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD MODEL ---
# We wrap this in a try-except block. 
# If 'model.h5' is not found, we switch to MOCK MODE for testing the UI.
model = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Warning: {MODEL_PATH} not found. Running in MOCK MODE.")
except ImportError:
    print("TensorFlow not installed. Running in MOCK MODE.")
except Exception as e:
    print(f"Error loading model: {e}. Running in MOCK MODE.")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, target_size=(224, 224)):
    """
    Prepares the image for the model (Resize, RGB, Normalize).
    Adjust target_size according to your specific model requirements.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize to [0, 1]
    return image

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            # Open image directly from memory
            image = Image.open(io.BytesIO(file.read()))
            
            # --- REAL PREDICTION LOGIC ---
            if model:
                # 1. Preprocess
                processed_image = preprocess_image(image)
                
                # 2. Predict
                prediction = model.predict(processed_image)
                
                # 3. Process Result
                # Assuming Model Output is Softmax: [Covid, Pneumonia, Normal]
                # Adjust indices based on how your specific model was trained
                class_names = ['positive', 'pneumonia', 'normal'] 
                
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                label_id = class_names[predicted_class_index]
                confidence = float(np.max(prediction) * 100)
                
                return jsonify({
                    'success': True,
                    'label_id': label_id,
                    'confidence': round(confidence, 2)
                })
            
            # --- MOCK PREDICTION LOGIC (For Testing UI) ---
            else:
                import random
                import time
                
                # Simulate processing time
                time.sleep(1) 
                
                # Randomly select a result to demonstrate UI capabilities
                mock_classes = ['positive', 'pneumonia', 'normal']
                mock_label = random.choice(mock_classes)
                mock_confidence = random.uniform(85.0, 99.9)
                
                return jsonify({
                    'success': True,
                    'label_id': mock_label,
                    'confidence': round(mock_confidence, 2),
                    'note': 'This is a MOCK result because model.h5 was not found.'
                })

        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({'error': 'Internal server error during prediction'})
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)