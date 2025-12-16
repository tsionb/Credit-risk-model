import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def test_metrics_calculation():
    """Test metrics calculation logic"""
    from train import CreditRiskModelTrainer
    
    trainer = CreditRiskModelTrainer(random_state=42)
    
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.3, 0.4, 0.2])
    
    metrics = trainer.calculate_metrics(y_true, y_pred, y_pred_proba)
    
    assert 'roc_auc' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 0 <= metrics['roc_auc'] <= 1
    
    print(" Metrics calculation test passed")
    return True

def test_model_file_exists():
    """Test that model file was created"""
    assert os.path.exists('models/best_model.pkl'), "Best model not saved"
    print(" Model file exists")
    

if __name__ == "__main__":
    test_metrics_calculation()
    test_model_file_exists()
    print(" All tests passed!")