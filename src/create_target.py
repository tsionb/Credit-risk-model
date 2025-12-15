import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class RFMClusterTarget:
    """Create high-risk labels using RFM clustering"""
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.rfm_features = None
        self.cluster_labels = None
        
    def calculate_rfm(self, features_df):
        """
        Calculate RFM metrics from features
        Handles missing CustomerId column
        """
        print("📊 Calculating RFM metrics...")
        
        # Create a copy to avoid modifying original
        df = features_df.copy()
        
        # Add customer index if no ID column exists
        if 'CustomerId' not in df.columns:
            df['CustomerId'] = range(len(df))
            print(" No CustomerId column found, using row indices")
        
        # Extract or calculate RFM features
        rfm_data = {}
        rfm_data['CustomerId'] = df['CustomerId'].values
        
        # 1. RECENCY: Look for recency-related columns
        recency_candidates = ['days_since_last_transaction', 'recency', 
                            'last_transaction_days', 'inactivity_days']
        recency_found = False
        
        for col in recency_candidates:
            if col in df.columns:
                rfm_data['recency'] = df[col].values
                print(f" Using '{col}' as recency")
                recency_found = True
                break
        
        if not recency_found:
            # Create placeholder (will be scaled anyway)
            rfm_data['recency'] = np.random.normal(0, 1, len(df))
            print(" No recency column found, using random values")
        
        # 2. FREQUENCY: Look for frequency-related columns
        freq_candidates = ['total_transactions', 'transaction_count', 
                          'frequency', 'count', 'num_transactions']
        freq_found = False
        
        for col in freq_candidates:
            if col in df.columns:
                rfm_data['frequency'] = df[col].values
                print(f" Using '{col}' as frequency")
                freq_found = True
                break
        
        if not freq_found:
            # Try to find any count column
            count_cols = [col for col in df.columns if 'count' in col.lower()]
            if count_cols:
                rfm_data['frequency'] = df[count_cols[0]].values
                print(f" Using '{count_cols[0]}' as frequency")
                freq_found = True
        
        if not freq_found:
            rfm_data['frequency'] = np.ones(len(df))
            print(" No frequency column found, using ones")
        
        # 3. MONETARY: Look for monetary-related columns
        monetary_candidates = ['total_amount', 'monetary', 'total_value', 
                             'amount_sum', 'value_sum', 'avg_amount', 'avg_value']
        monetary_found = False
        
        for col in monetary_candidates:
            if col in df.columns:
                rfm_data['monetary'] = np.abs(df[col].values)  # Use absolute value
                print(f" Using '{col}' as monetary")
                monetary_found = True
                break
        
        if not monetary_found:
            # Look for any amount/value column
            amount_cols = [col for col in df.columns 
                          if 'amount' in col.lower() or 'value' in col.lower()]
            if amount_cols:
                rfm_data['monetary'] = np.abs(df[amount_cols[0]].values)
                print(f" Using '{amount_cols[0]}' as monetary")
                monetary_found = True
        
        if not monetary_found:
            rfm_data['monetary'] = np.ones(len(df))
            print(" No monetary column found, using ones")
        
        # Create DataFrame
        rfm_df = pd.DataFrame(rfm_data)
        
        # Fill any NaN values
        rfm_df = rfm_df.fillna(rfm_df.median())
        
        print(f" RFM calculated for {len(rfm_df)} customers")
        print(f"   Columns: {list(rfm_df.columns)}")
        
        return rfm_df
    
    def create_clusters(self, features_df):
        """
        Create clusters and assign high-risk labels
        """
        # Calculate RFM
        rfm_df = self.calculate_rfm(features_df)
        self.rfm_features = rfm_df
        
        # Extract numeric RFM columns (exclude CustomerId)
        numeric_cols = [col for col in rfm_df.columns 
                       if col != 'CustomerId' and pd.api.types.is_numeric_dtype(rfm_df[col])]
        
        if len(numeric_cols) < 2:
            raise ValueError(f"Need at least 2 numeric RFM features, found {len(numeric_cols)}")
        
        rfm_numeric = rfm_df[numeric_cols]
        
        # Scale features
        print(f"⚖️ Scaling {len(numeric_cols)} RFM features...")
        rfm_scaled = self.scaler.fit_transform(rfm_numeric)
        
        # Apply K-means clustering
        print(f" Clustering with K-means (k={self.n_clusters})...")
        clusters = self.kmeans.fit_predict(rfm_scaled)
        
        # Add clusters to dataframe
        rfm_df['cluster'] = clusters
        self.cluster_labels = clusters
        
        # Analyze clusters to identify high-risk
        print(" Analyzing clusters for risk assessment...")
        
        # Calculate cluster statistics
        cluster_stats = rfm_numeric.copy()
        cluster_stats['cluster'] = clusters
        
        cluster_means = cluster_stats.groupby('cluster').mean()
        print("\n📈 Cluster Statistics (mean values):")
        print(cluster_means)
        
        # Identify high-risk cluster
        # Strategy: Cluster with highest recency, lowest frequency, lowest monetary
        risk_scores = {}
        
        for cluster in range(self.n_clusters):
            cluster_data = cluster_means.loc[cluster]
            
            # Normalize each metric (assuming higher values are worse for recency, 
            # better for frequency and monetary)
            scores = []
            
            if 'recency' in cluster_data.index:
                # Higher recency = higher risk
                scores.append(cluster_data['recency'])
            
            if 'frequency' in cluster_data.index:
                # Lower frequency = higher risk (invert)
                scores.append(-cluster_data['frequency'])
            
            if 'monetary' in cluster_data.index:
                # Lower monetary = higher risk (invert)
                scores.append(-cluster_data['monetary'])
            
            # Average the scores
            if scores:
                risk_scores[cluster] = np.mean(scores)
            else:
                # Fallback: use cluster size (smaller = potentially riskier)
                cluster_size = (clusters == cluster).sum()
                risk_scores[cluster] = -cluster_size  # Smaller clusters get higher score
        
        # Find highest risk cluster
        high_risk_cluster = max(risk_scores, key=risk_scores.get)
        print(f"\n High-risk cluster identified: Cluster {high_risk_cluster}")
        print(f"   Risk score: {risk_scores[high_risk_cluster]:.3f}")
        
        # Create binary target
        rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
        
        # Calculate risk distribution
        risk_counts = rfm_df['is_high_risk'].value_counts()
        print(f"\n📊 Risk Distribution:")
        print(f"   High-risk customers: {risk_counts.get(1, 0)} ({risk_counts.get(1, 0)/len(rfm_df)*100:.1f}%)")
        print(f"   Low-risk customers: {risk_counts.get(0, 0)} ({risk_counts.get(0, 0)/len(rfm_df)*100:.1f}%)")
        
        # Visualize clusters
        self._visualize_clusters(rfm_numeric, clusters, high_risk_cluster)
        
        return rfm_df[['CustomerId', 'is_high_risk', 'cluster']]
    
    def _visualize_clusters(self, rfm_data, clusters, high_risk_cluster):
        """Visualize RFM clusters"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Scatter plot of first two dimensions
            if rfm_data.shape[1] >= 2:
                plt.subplot(2, 2, 1)
                scatter = plt.scatter(rfm_data.iloc[:, 0], rfm_data.iloc[:, 1], 
                                     c=clusters, cmap='viridis', alpha=0.6, s=50)
                plt.xlabel(rfm_data.columns[0])
                plt.ylabel(rfm_data.columns[1])
                plt.title('RFM Clusters (First 2 Dimensions)')
                plt.colorbar(scatter, label='Cluster')
            
            # Plot 2: Risk distribution
            plt.subplot(2, 2, 2)
            risk_labels = ['Low Risk', 'High Risk']
            risk_counts = [(clusters != high_risk_cluster).sum(), 
                          (clusters == high_risk_cluster).sum()]
            colors = ['lightgreen', 'salmon']
            bars = plt.bar(risk_labels, risk_counts, color=colors, edgecolor='black')
            plt.title('Risk Distribution')
            plt.ylabel('Number of Customers')
            
            # Add count labels on bars
            for bar, count in zip(bars, risk_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_counts)*0.01, 
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Cluster sizes
            plt.subplot(2, 2, 3)
            cluster_sizes = [(clusters == i).sum() for i in range(self.n_clusters)]
            cluster_colors = ['salmon' if i == high_risk_cluster else 'lightblue' 
                             for i in range(self.n_clusters)]
            bars = plt.bar(range(self.n_clusters), cluster_sizes, color=cluster_colors, edgecolor='black')
            plt.title('Cluster Sizes')
            plt.xlabel('Cluster')
            plt.ylabel('Number of Customers')
            plt.xticks(range(self.n_clusters))
            
            # Add labels
            for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cluster_sizes)*0.01, 
                        f"{size}\n({'High' if i == high_risk_cluster else 'Low'})", 
                        ha='center', va='bottom', fontsize=9)
            
            # Plot 4: Box plot of RFM features by cluster
            plt.subplot(2, 2, 4)
            if rfm_data.shape[1] >= 1:
                feature_data = []
                cluster_labels = []
                feature_names = []
                
                # Take first feature for box plot
                feature = rfm_data.columns[0]
                for cluster in range(self.n_clusters):
                    feature_data.append(rfm_data.iloc[clusters == cluster, 0])
                    cluster_labels.append(f'Cluster {cluster}')
                    feature_names.append(feature)
                
                box = plt.boxplot(feature_data, labels=cluster_labels, patch_artist=True)
                
                # Color boxes
                for i, patch in enumerate(box['boxes']):
                    patch.set_facecolor('salmon' if i == high_risk_cluster else 'lightblue')
                
                plt.title(f'{feature} Distribution by Cluster')
                plt.ylabel(feature)
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs('reports', exist_ok=True)
            plot_path = 'reports/rfm_clusters.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Cluster visualization saved to {plot_path}")
            
        except Exception as e:
            print(f" Could not create visualization: {e}")
            import traceback
            traceback.print_exc()

def create_proxy_target(features_path, output_path=None):
    """
    Main function to create proxy target
    
    Parameters:
    -----------
    features_path : str
        Path to features CSV from Task 3
    output_path : str
        Path to save target labels
        
    Returns:
    --------
    target_df : DataFrame
        CustomerId and risk labels
    """
    print(" Creating proxy target variable...")
    print("="*50)
    
    # Load features
    features = pd.read_csv(features_path)
    print(f"📂 Loaded features: {features.shape}")
    print(f"📋 Sample columns: {list(features.columns[:10])}...")
    
    # Check if CustomerId exists
    if 'CustomerId' not in features.columns:
        print(" CustomerId column not found in features")
        print("   Creating sequential customer IDs...")
        features = features.copy()
        features['CustomerId'] = range(len(features))
    
    # Create target
    target_creator = RFMClusterTarget(n_clusters=3, random_state=42)
    target_df = target_creator.create_clusters(features)
    
    # Save target
    if output_path:
        target_df.to_csv(output_path, index=False)
        print(f"\n💾 Target saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("PROXY TARGET CREATION COMPLETE")
    print("="*50)
    print(f"Total customers: {len(target_df)}")
    print(f"High-risk customers: {target_df['is_high_risk'].sum()} ({target_df['is_high_risk'].mean()*100:.1f}%)")
    print(f"Low-risk customers: {len(target_df) - target_df['is_high_risk'].sum()} ({(1 - target_df['is_high_risk'].mean())*100:.1f}%)")
    
    return target_df

if __name__ == "__main__":
    # Example usage
    target_df = create_proxy_target(
        features_path='data/processed/features.csv',
        output_path='data/processed/target.csv'
    )
    print("\nFirst 10 target labels:")
    print(target_df.head(10))