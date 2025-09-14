# import os
# import numpy as np
# import tensorflow as tf
# from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename
# from tensorflow.keras.utils import load_img, img_to_array

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# # Create upload folder if it doesn't exist
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load the model
# model = None

# def load_model():
#     global model
#     try:
#         model = tf.keras.models.load_model('Crop_Disease_Detection.keras')
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Using a placeholder model for now...")
#         # Create a simple placeholder model for testing the UI
#         inputs = tf.keras.layers.Input(shape=(160, 160, 3))
#         x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
#         outputs = tf.keras.layers.Dense(39, activation='softmax')(x)
#         model = tf.keras.Model(inputs=inputs, outputs=outputs)

# # Load model when app starts
# with app.app_context():
#     load_model()

# # Class labels from the notebook
# class_labels = [
#     'Apple___Apple_scab',
#     'Apple___Black_rot',
#     'Apple___Cedar_apple_rust',
#     'Apple___healthy',
#     'Background_without_leaves',
#     'Blueberry___healthy',
#     'Cherry___Powdery_mildew',
#     'Cherry___healthy',
#     'Corn___Cercospora_leaf_spot Gray_leaf_spot',
#     'Corn___Common_rust',
#     'Corn___Northern_Leaf_Blight',
#     'Corn___healthy',
#     'Grape___Black_rot',
#     'Grape___Esca_(Black_Measles)',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#     'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)',
#     'Peach___Bacterial_spot',
#     'Peach___healthy',
#     'Pepper,_bell___Bacterial_spot',
#     'Pepper,_bell___healthy',
#     'Potato___Early_blight',
#     'Potato___Late_blight',
#     'Potato___healthy',
#     'Raspberry___healthy',
#     'Soybean___healthy',
#     'Squash___Powdery_mildew',
#     'Strawberry___Leaf_scorch',
#     'Strawberry___healthy',
#     'Tomato___Bacterial_spot',
#     'Tomato___Early_blight',
#     'Tomato___Late_blight',
#     'Tomato___Leaf_Mold',
#     'Tomato___Septoria_leaf_spot',
#     'Tomato___Spider_mites Two-spotted_spider_mite',
#     'Tomato___Target_Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#     'Tomato___Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]

# # Disease information dictionary for providing treatment recommendations
# disease_info = {
#     'Apple___Apple_scab': {
#         'description': 'Apple scab is a common disease of apple trees caused by the fungus Venturia inaequalis.',
#         'treatment': 'Apply fungicides early in the season. Prune affected branches and remove fallen leaves to reduce infection sources.'
#     },
#     'Apple___Black_rot': {
#         'description': 'Black rot is a fungal disease that affects apples, caused by Botryosphaeria obtusa.',
#         'treatment': 'Prune out diseased branches. Apply fungicides during the growing season. Remove mummified fruits.'
#     },
#     # Add more disease information as needed
#     'default': {
#         'description': 'Information about this specific disease is not available in our database.',
#         'treatment': 'Consult with a local agricultural extension service for specific treatment recommendations.'
#     }
# }

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# def preprocess_image(image_path):
#     # Use TensorFlow's functions to load and preprocess the image
#     # Explicitly specify color_mode='rgb' to ensure 3 channels
#     img = load_img(image_path, target_size=(160, 160), color_mode='rgb')
#     img_array = img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     return img_array

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
        
#         try:
#             # Preprocess the image
#             img_array = preprocess_image(file_path)
            
#             # Make prediction
#             global model
#             if model is None:
#                 load_model()
                
#             preds = model.predict(img_array)
#             class_idx = np.argmax(preds, axis=1)[0]
#             confidence = float(np.max(preds))
            
#             # Get the predicted class label
#             predicted_class = class_labels[class_idx]
            
#             # Get disease information
#             info = disease_info.get(predicted_class, disease_info['default'])
            
#             # Format the result
#             result = {
#                 'class': predicted_class,
#                 'confidence': round(confidence * 100, 2),
#                 'description': info['description'],
#                 'treatment': info['treatment']
#             }
            
#             return jsonify(result)
            
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500
    
#     return jsonify({'error': 'File type not allowed'}), 400

# if __name__ == '__main__':
#     app.run(debug=False)



# code 2

# app.py (Streamlit version; prediction logic preserved)
import os
import io
import re
import uuid
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# -------------------------
# Labels (unchanged)
# -------------------------
class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# -------------------------
# Disease info (unchanged)
# -------------------------
disease_info = {
    'Apple___Apple_scab': {
        'description': 'Apple scab is a common disease of apple trees caused by the fungus Venturia inaequalis.',
        'treatment': 'Apply fungicides early in the season. Prune affected branches and remove fallen leaves to reduce infection sources.'
    },
    'Apple___Black_rot': {
        'description': 'Black rot is a fungal disease that affects apples, caused by Botryosphaeria obtusa.',
        'treatment': 'Prune out diseased branches. Apply fungicides during the growing season. Remove mummified fruits.'
    },
    # Add more disease information as needed
    'default': {
        'description': 'Information about this specific disease is not available in our database.',
        'treatment': 'Consult with a local agricultural extension service for specific treatment recommendations.'
    }
}

# -------------------------
# Ensure uploads directory exists (keeps parity with your Flask logic)
# -------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# Model loader (cached for Streamlit)
# -------------------------
@st.cache_resource
def load_model_cached():
    """
    Try to load your saved model file 'Crop_Disease_Detection.keras'.
    If it fails, create the same placeholder model you used previously.
    """
    try:
        model = tf.keras.models.load_model('Crop_Disease_Detection.keras')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        # print similar to your original code
        print(f"Error loading model: {e}")
        print("Using a placeholder model for now...")
        # Create the placeholder model (same as your original)
        inputs = tf.keras.layers.Input(shape=(160, 160, 3))
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(len(class_labels), activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

model = load_model_cached()

# -------------------------
# Safe filename helper (no werkzeug required)
# -------------------------
def safe_filename(filename: str) -> str:
    if not filename:
        return f"{uuid.uuid4().hex}.jpg"
    name = os.path.basename(filename)
    name = name.replace(" ", "_")
    name = re.sub(r'[^A-Za-z0-9_.-]', '', name)
    if name == "":
        return f"{uuid.uuid4().hex}.jpg"
    return name

# -------------------------
# Preprocess the image robustly (ensures RGB 3 channels, exact 160x160)
# -------------------------
def preprocess_image_filebytes(file_bytes: bytes, target_size=(160, 160)):
    """
    Input: raw bytes of an image (from Streamlit uploader)
    Output: numpy array shape (1, 160, 160, 3) with values [0,1]
    """
    # open with PIL from bytes, convert to RGB (ensures 3 channels)
    img = Image.open(io.BytesIO(file_bytes))
    img = img.convert("RGB")  # THIS fixes the 1-channel (grayscale) problem
    # Choose an appropriate resampling filter
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        # Pillow older versions
        resample = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.ANTIALIAS
    img = img.resize(target_size, resample)
    arr = np.array(img).astype(np.float32) / 255.0
    # Ensure shape: (1, H, W, C)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------------
# Streamlit UI (keeps result format same as Flask: class, confidence, description, treatment)
# -------------------------
st.set_page_config(page_title="Crop Disease Detection", layout="centered")
st.title("🌱 Crop Disease Detection")
st.markdown("Upload a leaf image to detect disease and get treatment recommendations.")

uploaded_file = st.file_uploader("Upload an image (jpg / jpeg / png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show preview
    try:
        # streamlit's uploaded_file supports .getvalue() or .read()
        file_bytes = uploaded_file.read()
        # display using PIL to avoid accidental channel issues in Streamlit preview
        preview_img = Image.open(io.BytesIO(file_bytes))
        st.image(preview_img, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error reading uploaded image: {e}")
        file_bytes = None

    if st.button("🔍 Predict") and file_bytes is not None:
        # Save uploaded file into uploads folder (like your Flask logic)
        try:
            filename = safe_filename(getattr(uploaded_file, "name", None))
            file_path = os.path.join(UPLOAD_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
        except Exception as e:
            st.warning(f"Could not save uploaded file to disk: {e}")
            file_path = None

        # Preprocess and predict (same core logic as your Flask version)
        try:
            img_array = preprocess_image_filebytes(file_bytes)  # ensures shape (1,160,160,3)

            # Extra guard: if model expects different channel count, attempt fix (shouldn't be needed because of convert('RGB'))
            try:
                expected_channels = model.input_shape[-1]
            except Exception:
                expected_channels = None

            if expected_channels is not None and img_array.shape[-1] != expected_channels:
                # example: model expects 3 but got 1 — we already converted to RGB so this is unlikely,
                # but keep a safe fallback to repeat channels if needed.
                if img_array.shape[-1] == 1 and expected_channels == 3:
                    img_array = np.repeat(img_array, 3, axis=-1)
                else:
                    raise ValueError(f"Model expects {expected_channels} channels but image has {img_array.shape[-1]}")

            preds = model.predict(img_array)
            class_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds))

            predicted_class = class_labels[class_idx]
            info = disease_info.get(predicted_class, disease_info['default'])

            result = {
                'class': predicted_class,
                'confidence': round(confidence * 100, 2),
                'description': info['description'],
                'treatment': info['treatment']
            }

            # Show the result (same keys as your Flask JSON)
            st.success(f"Prediction: **{result['class']}** ({result['confidence']}% confidence)")
            st.subheader("📝 Description")
            st.write(result['description'])
            st.subheader("💊 Treatment")
            st.write(result['treatment'])
            st.markdown("---")
            st.json(result)

        except Exception as e:
            st.error(f"Prediction error: {e}")
