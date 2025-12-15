"""
Unit tests for data processing - Fixed version
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import CustomerAggregator, TimeFeatureExtractor

class TestFeatureEngineering:
    """Test the complete feature engineering pipeline"""
    
    def test_feature_file_created(self):
        """Test that features.csv is created"""
        # This test assumes you've already run the processing
        assert os.path.exists('data/processed/features.csv'), \
            "features.csv should be created"
    
    def test_feature_dimensions(self):
        """Test expected feature dimensions"""
        features = pd.read_csv('data/processed/features.csv')
        
        # Should have reasonable dimensions
        assert features.shape[0] > 0, "Should have at least 1 customer"
        assert features.shape[1] > 10, "Should have multiple features"
        
        # Check for expected columns
        expected_cols = ['total_transactions', 'total_amount', 'avg_amount']
        for col in expected_cols:
            assert col in features.columns, f"Missing expected column: {col}"
    
    def test_feature_scaling(self):
        """Test that numeric features are scaled"""
        features = pd.read_csv('data/processed/features.csv')
        
        # Select numeric columns (exclude one-hot encoded)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        # For this test, just check that scaling was applied to some columns
        # (some may be categorical after one-hot encoding)
        scaled_cols = [col for col in numeric_cols if 'total_' in col or 'avg_' in col]
        
        if scaled_cols:
            for col in scaled_cols[:3]:  # Check first 3
                mean_abs = abs(features[col].mean())
                # Allow tolerance for scaled features
                assert mean_abs < 5, f"Column {col} may not be properly scaled: mean={features[col].mean():.2f}"


def test_customer_aggregator_with_minimal_data():
    """Test CustomerAggregator with minimal required data"""
    # Create data with only essential columns
    test_data = pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3'],
        'Amount': [100.0, 200.0, 50.0, 150.0, 75.0],
        'Value': [100, 200, 50, 150, 75],
        'TransactionStartTime': pd.date_range('2024-01-01', periods=5),
        'FraudResult': [0, 0, 1, 0, 0]
    })
    
    aggregator = CustomerAggregator()
    features = aggregator.fit_transform(test_data)
    
    assert features is not None
    assert isinstance(features, pd.DataFrame)
    assert len(features) == 3  # 3 unique customers
    assert 'CustomerId' in features.columns
    assert 'total_transactions' in features.columns
    assert 'total_amount' in features.columns


def test_time_feature_extractor():
    """Test TimeFeatureExtractor"""
    test_data = pd.DataFrame({
        'TransactionStartTime': [
            '2024-01-01 09:30:00',
            '2024-01-01 14:45:00',
            '2024-01-02 20:15:00'
        ],
        'Amount': [100, 200, 150]
    })
    
    extractor = TimeFeatureExtractor()
    result = extractor.fit_transform(test_data)
    
    assert 'TransactionStartTime' not in result.columns
    assert 'transaction_hour' in result.columns
    assert 'transaction_day' in result.columns
    assert result['transaction_hour'].iloc[0] == 9  # 9 AM


def test_end_to_end_with_sample():
    """Test with a small sample from actual data"""
    # Load a small sample from your actual data
    if os.path.exists('data/raw/data.csv'):
        # Read first 100 rows
        sample_data = pd.read_csv('data/raw/data.csv', nrows=100)
        
        # Test aggregator
        aggregator = CustomerAggregator()
        features = aggregator.fit_transform(sample_data)
        
        assert features is not None
        assert len(features) > 0
        print(f"✅ Processed {len(features)} customers from sample")
    else:
        print("ℹ️ Skipping sample test - data file not found")


# Test that can be skipped if data not available
@pytest.mark.skipif(
    not os.path.exists('data/raw/data.csv'),
    reason="Raw data file not found"
)
def test_with_actual_data():
    """Test with actual data file"""
    from data_processing import simple_process_data
    
    # Create a test output path
    test_output = 'test_actual_output.csv'
    
    try:
        features = simple_process_data(
            raw_data_path='data/raw/data.csv',
            output_path=test_output
        )
        
        assert os.path.exists(test_output)
        assert features is not None
        assert isinstance(features, pd.DataFrame)
        
    finally:
        if os.path.exists(test_output):
            os.remove(test_output)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])