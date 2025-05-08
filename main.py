"""
Main Execution Script
Runs the complete pipeline from data to evaluation
"""
from data_preprocessing import load_data
from models.base_model import build_base_model
from train import train_model
from evaluate import evaluate_model