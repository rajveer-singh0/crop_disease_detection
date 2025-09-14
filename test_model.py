import tensorflow as tf
import numpy as np

# Try to load the model
try:
    print("Attempting to load model...")
    model = tf.keras.models.load_model('Crop_Disease_Detection.keras')
    print("Model loaded successfully!")
    
    # Print model summary
    model.summary()
    
    # Create a random test input with the correct shape
    test_input = np.random.random((1, 160, 160, 3)).astype('float32')
    print(f"Test input shape: {test_input.shape}")
    
    # Try a prediction
    prediction = model.predict(test_input)
    print(f"Prediction shape: {prediction.shape}")
    print("Model works correctly!")
    
except Exception as e:
    print(f"Error loading or using model: {e}")