import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  #  fixes Windows threading issue
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Import target creation
from create_target import create_proxy_target

class CreditRiskModelTrainer:
    """Main class for training credit risk models"""
    
    def __init__(self, experiment_name="credit_risk_modeling", random_state=42):
        self.random_state = random_state
        self.experiment_name = experiment_name
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.scaler = StandardScaler()
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        print(f" MLflow Experiment: {experiment_name}")
    
    def load_and_prepare_data(self, features_path, target_path=None, 
                             test_size=0.2, create_target=True):
        """
        Load features and target, prepare train/test split
        
        Parameters:
        -----------
        features_path : str
            Path to features CSV
        target_path : str
            Path to target CSV (if None and create_target=True, will create)
        test_size : float
            Test set proportion
        create_target : bool
            Whether to create target if not exists
        """
        print("📂 Loading data...")
        
        # Load features
        X = pd.read_csv(features_path)
        print(f"Features shape: {X.shape}")
        
        # Load or create target
        if target_path and os.path.exists(target_path):
            target_df = pd.read_csv(target_path)
            y = target_df['is_high_risk'].values
            print(f"Target loaded from {target_path}")
        elif create_target:
            print("Creating proxy target...")
            target_df = create_proxy_target(features_path)
            y = target_df['is_high_risk'].values
            if target_path:
                target_df.to_csv(target_path, index=False)
        else:
            raise ValueError("Target not found and create_target=False")
        
        # Check for CustomerId column and drop if present
        if 'CustomerId' in X.columns:
            X = X.drop('CustomerId', axis=1)
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Check for class imbalance
        class_counts = pd.Series(y).value_counts()
        print(f"\n📊 Class distribution:")
        for cls, count in class_counts.items():
            percentage = count / len(y) * 100
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
        
        # Handle severe imbalance
        if class_counts.min() / class_counts.max() < 0.1:
            print(" Severe class imbalance detected. Using class_weight='balanced'")
            self.use_class_weight = True
        else:
            self.use_class_weight = False
        
        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y  # Preserve class distribution
        )
        
        print(f"\n Data prepared:")
        print(f"  Training set: {self.X_train.shape}")
        print(f"  Test set: {self.X_test.shape}")
        print(f"  Features: {self.X_train.shape[1]}")
        
        return X, y
    
    def train_logistic_regression(self, param_grid=None):
        """Train and tune Logistic Regression"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        # Default parameters
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
        
        # Create model
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Grid search
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        # Train
        with mlflow.start_run(run_name="logistic_regression"):
            grid_search.fit(self.X_train, self.y_train)
            
            # Log parameters
            mlflow.log_params(grid_search.best_params_)
            
            # Evaluate
            y_pred = grid_search.predict(self.X_test)
            y_pred_proba = grid_search.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
            
            # Save visualization
            self._plot_roc_curve(self.y_test, y_pred_proba, "Logistic Regression")
            
            # Store model
            self.models['logistic_regression'] = {
                'model': grid_search.best_estimator_,
                'metrics': metrics,
                'best_params': grid_search.best_params_
            }
            
            print(f" Best parameters: {grid_search.best_params_}")
            print(f" Best ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Update best model
            if metrics['roc_auc'] > self.best_score:
                self.best_score = metrics['roc_auc']
                self.best_model = grid_search.best_estimator_
                self.best_model_name = "logistic_regression"
    
    def train_random_forest(self, param_grid=None):
        """Train and tune Random Forest"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        # Default parameters
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        
        # Create model
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Randomized search (faster for Random Forest)
        random_search = RandomizedSearchCV(
            rf, param_grid, n_iter=20, cv=3, scoring='roc_auc',
            n_jobs=-1, random_state=self.random_state, verbose=1
        )
        
        # Train
        with mlflow.start_run(run_name="random_forest"):
            random_search.fit(self.X_train, self.y_train)
            
            # Log parameters
            mlflow.log_params(random_search.best_params_)
            
            # Evaluate
            y_pred = random_search.predict(self.X_test)
            y_pred_proba = random_search.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(random_search.best_estimator_, "model")
            
            # Plot feature importance
            self._plot_feature_importance(
                random_search.best_estimator_, 
                self.X_train.columns,
                "Random Forest Feature Importance"
            )
            
            # Store model
            self.models['random_forest'] = {
                'model': random_search.best_estimator_,
                'metrics': metrics,
                'best_params': random_search.best_params_
            }
            
            print(f" Best parameters: {random_search.best_params_}")
            print(f" Best ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Update best model
            if metrics['roc_auc'] > self.best_score:
                self.best_score = metrics['roc_auc']
                self.best_model = random_search.best_estimator_
                self.best_model_name = "random_forest"
    
    def train_xgboost(self, param_grid=None):
        """Train XGBoost (optional)"""
        try:
            import xgboost as xgb
        except ImportError:
            print(" XGBoost not installed. Skipping XGBoost training.")
            return
        
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        # Default parameters
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        
        # Create model
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=10, cv=3, scoring='roc_auc',
            random_state=self.random_state, verbose=1
        )
        
        # Train
        with mlflow.start_run(run_name="xgboost"):
            random_search.fit(self.X_train, self.y_train)
            
            # Log parameters
            mlflow.log_params(random_search.best_params_)
            
            # Evaluate
            y_pred = random_search.predict(self.X_test)
            y_pred_proba = random_search.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(random_search.best_estimator_, "model")
            
            # Store model
            self.models['xgboost'] = {
                'model': random_search.best_estimator_,
                'metrics': metrics,
                'best_params': random_search.best_params_
            }
            
            print(f" Best parameters: {random_search.best_params_}")
            print(f" Best ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Update best model
            if metrics['roc_auc'] > self.best_score:
                self.best_score = metrics['roc_auc']
                self.best_model = random_search.best_estimator_
                self.best_model_name = "xgboost"
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'positive_rate': y_pred.mean()
        }
    
    def _plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Plot and save ROC curve"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save plot
            os.makedirs('reports', exist_ok=True)
            plot_path = f'reports/roc_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(plot_path)
            print(f" ROC curve saved: {plot_path}")
            
        except Exception as e:
            print(f" Could not create ROC curve: {e}")
    
    def _plot_feature_importance(self, model, feature_names, title):
        """Plot feature importance - Thread-safe version"""
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:]  # Top 15 features
            
            plt.figure(figsize=(10, 6))
            plt.title(title)
            bars = plt.barh(range(len(indices)), importances[indices], color='steelblue', align='center')
            plt.xlabel('Relative Importance')
            plt.ylabel('Features')
            
            # Use actual feature names if available
            if feature_names is not None:
                # Try to get readable names
                try:
                    feature_labels = [str(feature_names[i]) for i in indices]
                except:
                    feature_labels = [f'Feature {i}' for i in indices]
                plt.yticks(range(len(indices)), feature_labels)
            else:
                plt.yticks(range(len(indices)), indices)
            
            # Add value labels on bars
            for i, (bar, idx) in enumerate(zip(bars, indices)):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importances[idx]:.4f}', va='center')
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs('reports', exist_ok=True)
            plot_path = 'reports/feature_importance.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(plot_path)
            print(f" Feature importance saved: {plot_path}")
            
        except Exception as e:
            print(f" Could not plot feature importance: {e}")
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        if not self.models:
            print("No models trained yet!")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, model_info in self.models.items():
            metrics = model_info['metrics']
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model_name = max(self.models.items(), 
                            key=lambda x: x[1]['metrics']['roc_auc'])[0]
        print(f"\n Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"   ROC-AUC: {self.models[best_model_name]['metrics']['roc_auc']:.4f}")
        
        # Save comparison
        os.makedirs('reports', exist_ok=True)
        comparison_df.to_csv('reports/model_comparison.csv', index=False)
        
        return comparison_df
    
    def save_best_model(self, output_path='models/best_model.pkl'):
        """Save the best model"""
        if self.best_model is None:
            print(" No best model to save. Train models first.")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.best_model, output_path)
        print(f" Best model saved to: {output_path}")
        
        # Also save model info
        model_info = {
            'model_name': self.best_model_name,
            'score': self.best_score,
            'timestamp': datetime.now().isoformat(),
            'features_count': self.X_train.shape[1]
        }
        joblib.dump(model_info, output_path.replace('.pkl', '_info.pkl'))

def main():
    """Main training pipeline"""
    print("🎯 CREDIT RISK MODEL TRAINING - TASK 5")
    print("="*60)
    
    # Initialize trainer
    trainer = CreditRiskModelTrainer(
        experiment_name="credit_risk_week4",
        random_state=42
    )
    
    # Step 1: Load and prepare data
    X, y = trainer.load_and_prepare_data(
        features_path='data/processed/features.csv',
        target_path='data/processed/target.csv',
        test_size=0.2,
        create_target=False
    )
    
    # Step 2: Train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    # Train Logistic Regression
    trainer.train_logistic_regression()
    
    # Train Random Forest
    trainer.train_random_forest()
    
    # Optional: Train XGBoost
    trainer.train_xgboost()
    
    # Step 3: Compare models
    comparison_df = trainer.compare_models()
    
    # Step 4: Save best model
    trainer.save_best_model()
    
    # Step 5: Generate final report
    print("\n" + "="*60)
    print("="*60)
    print(" Proxy target created")
    print(" Models trained and tuned")
    print(" Experiments tracked in MLflow")
    print(" Best model saved")
    print(" Evaluation complete")
    
    
    return trainer

if __name__ == "__main__":
    trainer = main()