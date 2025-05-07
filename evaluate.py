"""
Model Evaluation Module
Calculates metrics and generates reports
"""
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model(model, x_test, y_test):
    """
    Evaluate model performance on test set
    
    Args:
        model: Trained Keras model
        x_test, y_test: Test data
    
    Returns:
        Dictionary containing metrics
    """
    # Predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_test.flatten())
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }