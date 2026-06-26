# NeuroScan — Brain Tumor Detection Web App
Flask + TensorFlow CNN for brain MRI classification.

## Project Structure
```
brain_tumor_app/
├── app.py                      # Flask backend
├── brain_tumor_cnn_v2.h5       # Your trained model (place here)
├── requirements.txt
├── templates/
│   └── index.html              # Frontend UI
└── static/
    └── uploads/                # Uploaded images stored here
```

## Setup & Run

### 1. Place your trained model
Copy `brain_tumor_cnn_v2.h5` (saved from your notebook) into this folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

## How it works
1. User uploads a brain MRI image (JPG/PNG)
2. Flask receives it via POST `/predict`
3. Image is resized to 128×128, normalized, and passed to the CNN
4. Model returns probabilities for Normal vs Tumor
5. Result is displayed with confidence bar and probability breakdown

## Notes
- Max upload size: 10MB
- Supported formats: JPG, JPEG, PNG
- The model expects MRI scans similar to the training data (grayscale-like brain scans)
