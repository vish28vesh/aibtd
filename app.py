import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Config
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'brain_tumor_cnn_v2.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# Load model once at startup
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (128, 128))
    image_normalized = image_resized / 255.0
    image_input = np.reshape(image_normalized, (1, 128, 128, 3))
    return image_input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG or PNG image.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image_input = preprocess_image(filepath)
        prediction = model.predict(image_input)

        prob_normal = float(prediction[0][0])
        prob_tumor = float(prediction[0][1])
        pred_label = int(np.argmax(prediction))
        confidence = max(prob_normal, prob_tumor) * 100

        result = {
            'label': 'Tumor Detected' if pred_label == 1 else 'No Tumor Found',
            'is_tumor': pred_label == 1,
            'confidence': round(confidence, 2),
            'prob_normal': round(prob_normal * 100, 2),
            'prob_tumor': round(prob_tumor * 100, 2),
            'image_url': f'/static/uploads/{filename}'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
