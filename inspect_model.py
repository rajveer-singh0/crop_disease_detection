import tensorflow as tf
import numpy as np

try:
    # Load the model
    print("Loading model...")
    model = tf.keras.models.load_model('Crop_Disease_Detection.keras', compile=False)
    
    # Print model input details
    print("\nModel input details:")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            print(f"Input layer: {layer.name}, Input shape: {layer.input_shape}")
    
    # Print first few layers
    print("\nFirst few layers:")
    for i, layer in enumerate(model.layers[:5]):
        print(f"Layer {i}: {layer.name}, Input shape: {layer.input_shape}, Output shape: {layer.output_shape}")
    
    # Create a new model with the correct input shape
    print("\nCreating a modified model...")
    inputs = tf.keras.layers.Input(shape=(160, 160, 3))
    x = inputs
    
    # Skip the input layer of the original model
    for layer in model.layers[1:]:  # Skip the input layer
        x = layer(x)
    
    new_model = tf.keras.Model(inputs=inputs, outputs=x)
    
    # Save the modified model
    new_model.save('Modified_Crop_Disease_Detection.keras')
    print("Modified model saved successfully!")
    
except Exception as e:
    print(f"Error: {e}")