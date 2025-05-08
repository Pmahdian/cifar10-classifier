"""
Main Execution Script
Runs the complete pipeline from data to evaluation
"""
from data_preprocessing import load_data
from models.base_model import build_base_model
from models.train import train_model
from evaluate import evaluate_model

def main():
    # Load and preprocess data
    print("Loading data...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    
    # Build model
    print("Building model...")
    model = build_base_model()
    
    # Train model
    print("Training model...")
    history = train_model(model, x_train, y_train, x_val, y_val)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, x_test, y_test)
    
    print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
