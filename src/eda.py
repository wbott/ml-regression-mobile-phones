import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ExploratoryDataAnalysis:
    def __init__(self, df):
        self.df = df
        
    def basic_info(self):
        """Display basic information about the dataset"""
        print("Dataset Info:")
        print(self.df.info())
        print("\nDataset Description:")
        print(self.df.describe())
        return self.df.describe()
    
    def plot_target_distribution(self, target_col='Price', figsize=(15, 5)):
        """Plot distribution of target variable before and after transformation"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Box plot
        sns.boxplot(x=self.df[target_col], ax=axes[0], color="skyblue")
        axes[0].set_title(f"Box Plot of {target_col}")
        
        # Histogram
        axes[1].hist(self.df[target_col], bins=30, color="lightgreen", edgecolor="black")
        axes[1].set_title(f"Histogram of {target_col}")
        axes[1].set_xlabel(target_col)
        axes[1].set_ylabel("Frequency")
        
        # KDE plot
        sns.kdeplot(self.df[target_col], shade=True, ax=axes[2], color="coral")
        axes[2].set_title(f"KDE Plot of {target_col}")
        axes[2].set_xlabel(target_col)
        axes[2].set_ylabel("Density")
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self, figsize=(10, 8)):
        """Calculate and visualize correlation matrix"""
        corr = self.df.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()
        
        return corr
    
    def calculate_vif(self, target_col='Price'):
        """Calculate Variance Inflation Factor for multicollinearity detection"""
        df_numeric = self.df.select_dtypes(include=[float, int])
        df_features = df_numeric.drop(columns=[target_col])
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = df_features.columns
        vif_data["VIF"] = [
            variance_inflation_factor(df_features.values, i) 
            for i in range(len(df_features.columns))
        ]
        
        print("Variance Inflation Factor (VIF):")
        print(vif_data)
        print("\nVIF Interpretation:")
        print("VIF = 1: No multicollinearity")
        print("1 < VIF ≤ 5: Low to moderate multicollinearity")
        print("VIF > 5: Moderate multicollinearity")
        print("VIF > 10: High multicollinearity")
        
        return vif_data
    
    def feature_target_correlation(self, target_col='Price'):
        """Analyze correlation between features and target"""
        corr_with_target = self.df.corr()[target_col].sort_values(ascending=False)
        
        print(f"Correlation with {target_col}:")
        print(corr_with_target)
        
        # Create interpretation table
        interpretation = []
        for feature, corr_val in corr_with_target.items():
            if feature != target_col:
                if abs(corr_val) > 0.7:
                    strength = "Strong"
                elif abs(corr_val) > 0.4:
                    strength = "Moderate"
                elif abs(corr_val) > 0.2:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                
                direction = "Positive" if corr_val > 0 else "Negative"
                interpretation.append({
                    'Feature': feature,
                    'Correlation': f"{corr_val:.2f}",
                    'Strength': strength,
                    'Direction': direction
                })
        
        interpretation_df = pd.DataFrame(interpretation)
        print(f"\nCorrelation Interpretation:")
        print(interpretation_df)
        
        return corr_with_target, interpretation_df
    
    def plot_feature_distributions(self, figsize=(15, 10)):
        """Plot distributions of all features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i].hist(self.df[col], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_eda(self):
        """Run comprehensive exploratory data analysis"""
        print("="*50)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        stats = self.basic_info()
        
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        self.plot_target_distribution()
        
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        corr_matrix = self.correlation_analysis()
        
        print("\n" + "="*50)
        print("FEATURE-TARGET CORRELATION")
        print("="*50)
        corr_with_target, interpretation = self.feature_target_correlation()
        
        print("\n" + "="*50)
        print("MULTICOLLINEARITY ANALYSIS")
        print("="*50)
        vif_data = self.calculate_vif()
        
        print("\n" + "="*50)
        print("FEATURE DISTRIBUTIONS")
        print("="*50)
        self.plot_feature_distributions()
        
        return {
            'basic_stats': stats,
            'correlation_matrix': corr_matrix,
            'feature_target_correlation': corr_with_target,
            'correlation_interpretation': interpretation,
            'vif_analysis': vif_data
        }