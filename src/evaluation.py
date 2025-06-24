import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
    
    def plot_actual_vs_predicted(self, y_true, y_pred, model_name, figsize=(8, 6)):
        """Create scatter plot of actual vs predicted values"""
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        sns.scatterplot(x=y_true, y=y_pred, color='blue', s=50, alpha=0.6)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 
                color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
        
        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        
        # Labels and title
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted - {model_name}\n(R² = {r2:.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        plt.plot(y_true, p(y_true), "g--", alpha=0.8, label=f'Trend Line (slope={z[0]:.3f})')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, model_name, figsize=(12, 4)):
        """Plot residual analysis"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot of Residuals')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return residuals
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mean_actual = np.mean(y_true)
        mean_predicted = np.mean(y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Adjusted R²
        n = len(y_true)
        p = 1  # Assuming single feature for simplicity
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        metrics = {
            'model_name': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'adjusted_r2': adj_r2,
            'mape': mape,
            'mean_actual': mean_actual,
            'mean_predicted': mean_predicted,
            'residual_std': np.std(y_true - y_pred)
        }
        
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def print_evaluation_summary(self, metrics):
        """Print formatted evaluation summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY - {metrics['model_name'].upper()}")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE):     {metrics['mae']:.6f}")
        print(f"Mean Squared Error (MSE):      {metrics['mse']:.6f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
        print(f"R-squared (R²):                {metrics['r2']:.6f}")
        print(f"Adjusted R²:                   {metrics['adjusted_r2']:.6f}")
        print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        print(f"Residual Standard Deviation:   {metrics['residual_std']:.6f}")
        print(f"Mean Actual Value:             {metrics['mean_actual']:.6f}")
        print(f"Mean Predicted Value:          {metrics['mean_predicted']:.6f}")
    
    def comprehensive_evaluation(self, y_true, y_pred, model_name):
        """Perform comprehensive model evaluation"""
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        
        # Print summary
        self.print_evaluation_summary(metrics)
        
        # Create visualizations
        self.plot_actual_vs_predicted(y_true, y_pred, model_name)
        residuals = self.plot_residuals(y_true, y_pred, model_name)
        
        return metrics, residuals
    
    def compare_models_visualization(self, models_data, figsize=(15, 10)):
        """Create comprehensive model comparison visualization"""
        n_models = len(models_data)
        
        fig, axes = plt.subplots(2, n_models, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, data) in enumerate(models_data.items()):
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            # Actual vs Predicted plot
            axes[0, i].scatter(y_true, y_pred, alpha=0.6, s=30)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            axes[0, i].set_xlabel('Actual')
            axes[0, i].set_ylabel('Predicted')
            axes[0, i].set_title(f'{model_name}\nR² = {r2_score(y_true, y_pred):.4f}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Residuals plot
            residuals = y_true - y_pred
            axes[1, i].scatter(y_pred, residuals, alpha=0.6, s=30)
            axes[1, i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Residuals')
            axes[1, i].set_title(f'Residuals - {model_name}')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle('Model Comparison: Actual vs Predicted & Residuals', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_metrics_comparison_table(self):
        """Create a formatted comparison table of all evaluated models"""
        if not self.evaluation_results:
            print("No models have been evaluated yet.")
            return None
        
        comparison_data = []
        for model_name, metrics in self.evaluation_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'MAE': metrics['mae'],
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'Adj. R²': metrics['adjusted_r2'],
                'MAPE (%)': metrics['mape']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('R²', ascending=False)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.6f'))
        
        # Highlight best model
        best_model = comparison_df.iloc[0]['Model']
        best_r2 = comparison_df.iloc[0]['R²']
        
        print(f"\n🏆 Best performing model: {best_model} (R² = {best_r2:.6f})")
        
        return comparison_df
    
    def model_performance_summary(self):
        """Generate a summary of model performance insights"""
        if not self.evaluation_results:
            print("No models have been evaluated yet.")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE INSIGHTS")
        print("="*60)
        
        # Find best and worst models
        r2_scores = {name: metrics['r2'] for name, metrics in self.evaluation_results.items()}
        best_model = max(r2_scores, key=r2_scores.get)
        worst_model = min(r2_scores, key=r2_scores.get)
        
        print(f"🏆 Best Model: {best_model} (R² = {r2_scores[best_model]:.6f})")
        print(f"📉 Worst Model: {worst_model} (R² = {r2_scores[worst_model]:.6f})")
        
        # Performance categorization
        for model_name, metrics in self.evaluation_results.items():
            r2 = metrics['r2']
            if r2 > 0.9:
                performance = "Excellent"
            elif r2 > 0.8:
                performance = "Very Good"
            elif r2 > 0.7:
                performance = "Good"
            elif r2 > 0.6:
                performance = "Fair"
            else:
                performance = "Poor"
            
            print(f"• {model_name}: {performance} (R² = {r2:.6f})")
        
        return best_model, worst_model