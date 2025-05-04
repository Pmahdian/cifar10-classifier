"""
CIFAR-10 Data Preprocessing Module
Handles data loading, splitting, and normalization
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split

def load_data(test_size=0.15, val_size=0.2):
    """
    Load and split CIFAR-10 dataset
    
    Args:
        test_size: Proportion for test set (default: 0.15)
        val_size: Proportion of validation from remaining data (default: 0.2)
    
    Returns:
        Six numpy arrays: x_train, x_val, x_test, y_train, y_val, y_test
    """
    # Load original data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Merge original train and test for custom split
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    
    # Split into train+val and test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42)
    
    # Split train_val into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_size, random_state=42)
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    return x_train, x_val, x_test, y_train, y_val, y_test