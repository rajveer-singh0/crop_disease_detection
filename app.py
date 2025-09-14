import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model('Crop_Disease_Detection.keras')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using a placeholder model for now...")
        # Create a simple placeholder model for testing the UI
        inputs = tf.keras.layers.Input(shape=(160, 160, 3))
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(39, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Load model when app starts
with app.app_context():
    load_model()

# Class labels from the notebook
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

# Disease information dictionary for providing treatment recommendations
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    # Use TensorFlow's functions to load and preprocess the image
    # Explicitly specify color_mode='rgb' to ensure 3 channels
    img = load_img(image_path, target_size=(160, 160), color_mode='rgb')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Preprocess the image
            img_array = preprocess_image(file_path)
            
            # Make prediction
            global model
            if model is None:
                load_model()
                
            preds = model.predict(img_array)
            class_idx = np.argmax(preds, axis=1)[0]
            confidence = float(np.max(preds))
            
            # Get the predicted class label
            predicted_class = class_labels[class_idx]
            
            # Get disease information
            info = disease_info.get(predicted_class, disease_info['default'])
            
            # Format the result
            result = {
                'class': predicted_class,
                'confidence': round(confidence * 100, 2),
                'description': info['description'],
                'treatment': info['treatment']
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))