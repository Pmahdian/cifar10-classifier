"""
Base CNN Model for CIFAR-10 Classification
Architecture: Conv -> Conv -> Pool -> Dropout -> Dense
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_base_model(input_shape=(32,32,3), num_classes=10):
    """
    Construct the baseline CNN model
    
    Args:
        input_shape: Shape of input images (default: 32,32,3)
        num_classes: Number of output classes (default: 10)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model