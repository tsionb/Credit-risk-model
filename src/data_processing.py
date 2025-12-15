"""
Feature Engineering Pipeline for Credit Risk Model
Transforms raw transaction data into model-ready features
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from xverse.transformer import WOE

# CUSTOM TRANSFORMERS

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Create customer-level aggregate features from transaction data"""
    
    def __init__(self, customer_id_col='CustomerId'):
        self.customer_id_col = customer_id_col
        self.aggregated_features = None
        
    def fit(self, X, y=None):
        # Calculate aggregates for fitting
        self._calculate_aggregates(X)
        return self
    
    def transform(self, X):
        # Return the aggregated features
        return self.aggregated_features
    
    def _calculate_aggregates(self, X):
        """Calculate various aggregate features per customer"""
        
        # Convert TransactionStartTime to datetime if not already
        if 'TransactionStartTime' in X.columns:
            X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Group by customer
        customer_groups = X.groupby(self.customer_id_col)
        
        # BASIC AGGREGATES
        agg_dict = {
            # Transaction counts
            'total_transactions': ('TransactionId', 'count'),
            'unique_days': ('TransactionStartTime', lambda x: x.dt.date.nunique()),
            
            # Amount aggregates
            'total_amount': ('Amount', 'sum'),
            'avg_amount': ('Amount', 'mean'),
            'median_amount': ('Amount', 'median'),
            'min_amount': ('Amount', 'min'),
            'max_amount': ('Amount', 'max'),
            'std_amount': ('Amount', 'std'),
            
            # Value aggregates (absolute amount)
            'total_value': ('Value', 'sum'),
            'avg_value': ('Value', 'mean'),
            
            # Behavioral metrics
            'avg_transactions_per_day': ('TransactionStartTime', lambda x: len(x) / x.dt.date.nunique() if x.dt.date.nunique() > 0 else 0),
            
            # Negative amount indicators (potential refunds/credits)
            'negative_transaction_count': ('Amount', lambda x: (x < 0).sum()),
            'negative_amount_ratio': ('Amount', lambda x: (x < 0).sum() / len(x) if len(x) > 0 else 0),
            
            # Fraud indicators
            'fraud_count': ('FraudResult', 'sum'),
            'fraud_ratio': ('FraudResult', 'mean'),
        }
        
        # Calculate aggregates
        aggregates = customer_groups.agg(**agg_dict)
        
        # RFM FEATURES
        if 'TransactionStartTime' in X.columns:
            # Recency: Days since last transaction
            latest_date = X['TransactionStartTime'].max()
            aggregates['days_since_last_transaction'] = customer_groups['TransactionStartTime'].max().apply(
                lambda x: (latest_date - x).days
            )
            
            # Frequency: Transactions per month (assuming data spans multiple months)
            date_range = (X['TransactionStartTime'].max() - X['TransactionStartTime'].min()).days
            if date_range > 30:
                aggregates['transactions_per_month'] = aggregates['total_transactions'] / (date_range / 30)
        
        # PRODUCT/CATEGORY PREFERENCES
        # Most frequent product category
        if 'ProductCategory' in X.columns:
            most_common_category = customer_groups['ProductCategory'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
            aggregates['preferred_category'] = most_common_category
        
        # CHANNEL PREFERENCES
        if 'ChannelId' in X.columns:
            most_common_channel = customer_groups['ChannelId'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'unknown')
            aggregates['preferred_channel'] = most_common_channel
        
        # Fill NaN values created by std (for customers with single transaction)
        aggregates = aggregates.fillna(0)
        
        self.aggregated_features = aggregates.reset_index()
        return self


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract time-based features from TransactionStartTime"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if 'TransactionStartTime' not in X.columns:
            return X
        
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        
        # Extract time features
        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        X['transaction_day'] = X['TransactionStartTime'].dt.day
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_year'] = X['TransactionStartTime'].dt.year
        X['transaction_dayofweek'] = X['TransactionStartTime'].dt.dayofweek
        X['transaction_weekend'] = X['transaction_dayofweek'].isin([5, 6]).astype(int)
        
        # Time of day categories
        X['time_of_day'] = pd.cut(
            X['transaction_hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        # Drop original column
        X = X.drop('TransactionStartTime', axis=1)
        
        return X


# MAIN PIPELINE

def create_feature_pipeline(target_col=None):
    """
    Create complete feature engineering pipeline
    """
    
    # Main pipeline steps
    steps = [
        # Step 1: Customer aggregation
        ('customer_aggregation', CustomerAggregator()),
        
        # Step 2: Time feature extraction
        ('time_features', TimeFeatureExtractor()),
    ]
    
    # Create initial pipeline
    initial_pipeline = Pipeline(steps)
    
    # We'll add column transformer after seeing the data
    return initial_pipeline


def process_data(raw_data_path, output_path=None, target_col=None):
    """
    Main function to process raw data through the pipeline
    """
    
    # Load raw data
    print(f"📂 Loading data from {raw_data_path}")
    raw_data = pd.read_csv(raw_data_path)
    
    # Separate target if provided
    X = raw_data.copy()
    y = None
    if target_col and target_col in X.columns:
        y = X[target_col]
        X = X.drop(target_col, axis=1)
    
    # Create initial pipeline
    print("🔧 Creating feature engineering pipeline...")
    pipeline = create_feature_pipeline(target_col)
    
    # Apply initial transformations
    print("🔄 Applying customer aggregation and time feature extraction...")
    X_transformed = pipeline.fit_transform(X, y)
    
    # Now handle the transformed data
    print("🔧 Creating column-aware preprocessing...")
    
    # Identify numeric and categorical columns
    X_df = pd.DataFrame(X_transformed) if isinstance(X_transformed, np.ndarray) else X_transformed
    
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"📊 Found {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns")
    
    # Create column transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Apply preprocessing
    print("🔄 Applying preprocessing (imputation, encoding, scaling)...")
    processed_data = preprocessor.fit_transform(X_df, y)
    
    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        # Fallback for older sklearn versions
        numeric_names = [f'num_{col}' for col in numeric_cols]
        categorical_names = []
        if categorical_cols:
            # Estimate one-hot encoded names
            for col in categorical_cols:
                unique_vals = X_df[col].nunique()
                for i in range(min(unique_vals, 10)):  # Limit to first 10 categories
                    categorical_names.append(f'cat_{col}_{i}')
        feature_names = numeric_names + categorical_names
    
    # Convert to DataFrame
    processed_data = pd.DataFrame(processed_data, columns=feature_names)
    
    # Save processed data
    if output_path:
        processed_data.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to {output_path}")
    
    print(f" Processing complete! Shape: {processed_data.shape}")
    
    # Combine pipelines for return
    full_pipeline = Pipeline([
        ('initial_transform', pipeline),
        ('preprocess', preprocessor)
    ])
    
    return processed_data, full_pipeline


# SIMPLIFIED VERSION

def simple_process_data(raw_data_path, output_path=None):
    """
    Simplified version that works step-by-step
    """
    print(f"📂 Loading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    print(f"📊 Original data shape: {df.shape}")
    
    # Step 1: Customer aggregation
    print("🔄 Step 1: Creating customer aggregates...")
    aggregator = CustomerAggregator()
    customer_features = aggregator.fit_transform(df)
    print(f"   Customer features shape: {customer_features.shape}")
    
    # Step 2: Time features
    print("🔄 Step 2: Extracting time features...")
    time_extractor = TimeFeatureExtractor()
    
    # We need to apply time extraction to original data for each transaction
    # For simplicity, let's extract time features at transaction level first
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    
    # Then aggregate time patterns per customer
    time_patterns = df.groupby('CustomerId').agg({
        'transaction_hour': ['mean', 'std'],
        'transaction_dayofweek': ['mean', 'std'],
        'transaction_month': ['nunique']
    })
    time_patterns.columns = ['_'.join(col).strip() for col in time_patterns.columns]
    time_patterns = time_patterns.reset_index()
    
    # Merge with customer features
    print("🔄 Step 3: Merging all features...")
    all_features = pd.merge(customer_features, time_patterns, on='CustomerId', how='left')
    
    # Handle missing values
    all_features = all_features.fillna(0)
    
    # Separate numeric and categorical columns
    numeric_cols = all_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = all_features.select_dtypes(include=['object']).columns.tolist()
    
    print(f"📊 Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"📊 Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Scale numeric columns
    print("🔄 Step 4: Scaling numeric features...")
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(all_features[numeric_cols])
    all_features[numeric_cols] = scaled_numeric
    
    # One-hot encode categorical columns
    if categorical_cols:
        print("🔄 Step 5: Encoding categorical features...")
        all_features = pd.get_dummies(all_features, columns=categorical_cols, drop_first=True)
    
    # Save results
    if output_path:
        all_features.to_csv(output_path, index=False)
        print(f"💾 Saved to {output_path}")
    
    print(f" Final shape: {all_features.shape}")
    return all_features


# EXAMPLE USAGE

if __name__ == "__main__":
    # Example usage
    import os
    
    # Paths
    raw_data_path = "./data/raw/data.csv"
    processed_data_path = "./data/processed/features.csv"
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    try:
        print("🚀 Starting feature engineering...")
        
        # Use the simplified version which is more robust
        processed_data = simple_process_data(
            raw_data_path=raw_data_path,
            output_path=processed_data_path
        )
        
        # Show sample of processed data
        print("\n📊 Sample of processed data (first 5 rows, first 10 columns):")
        print(processed_data.iloc[:5, :10])
        
        # Show summary
        print("\n" + "="*50)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*50)
        print(f"Total features created: {processed_data.shape[1]}")
        print(f"Total customers: {processed_data.shape[0]}")
        
        # Show feature types
        numeric_count = processed_data.select_dtypes(include=[np.number]).shape[1]
        print(f"Numeric features: {numeric_count}")
        
        print("\n📋 First 20 feature names:")
        for i, col in enumerate(processed_data.columns[:20]):
            print(f"  {i+1:2d}. {col}")
        
        if len(processed_data.columns) > 20:
            print(f"  ... and {len(processed_data.columns) - 20} more features")
            
    except FileNotFoundError:
        print(f" File not found: {raw_data_path}")
        print("Please check the path to your raw data file.")
    except Exception as e:
        print(f" Error processing data: {e}")
        import traceback
        traceback.print_exc()