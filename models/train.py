"""
Model Training Script
Handles training process with callbacks and logging
"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, x_train, y_train, x_val, y_val,
               epochs=50, batch_size=64, patience=5):
    """
    Train model with early stopping and checkpointing
    
    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_val, y_val: Validation data
        epochs: Maximum training epochs (default: 50)
        batch_size: Batch size (default: 64)
        patience: Early stopping patience (default: 5)
    
    Returns:
        Training history object
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return history