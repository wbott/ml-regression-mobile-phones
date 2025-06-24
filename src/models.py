import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_ols_model(self, X_train, y_train, feature_names=None):
        """Train Ordinary Least Squares model using statsmodels"""
        # Add constant term for intercept
        X_train_const = sm.add_constant(X_train)
        
        # Fit OLS model
        ols_model = sm.OLS(y_train, X_train_const).fit()
        
        self.models['ols'] = ols_model
        
        print("OLS Model Summary:")
        print(ols_model.summary())
        
        return ols_model
    
    def train_linear_regression(self, X_train, y_train):
        """Train Linear Regression model using sklearn"""
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        self.models['linear_regression'] = lr_model
        
        return lr_model
    
    def train_ridge_regression(self, X_train, y_train, alpha=1.0):
        """Train Ridge Regression model to handle multicollinearity"""
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train, y_train)
        
        self.models['ridge'] = ridge_model
        
        print(f"Ridge Regression trained with alpha={alpha}")
        
        return ridge_model
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, random_state=42):
        """Train Random Forest Regressor"""
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf_model.fit(X_train, y_train)
        
        self.models['random_forest'] = rf_model
        
        print(f"Random Forest trained with {n_estimators} trees")
        
        return rf_model
    
    def make_predictions(self, model_name, X_test):
        """Make predictions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if model_name == 'ols':
            # Add constant for OLS predictions
            X_test_const = sm.add_constant(X_test)
            predictions = model.predict(X_test_const)
        else:
            predictions = model.predict(X_test)
        
        return predictions
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.make_predictions(model_name, X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mse)
        
        results = {
            'model_name': model_name,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name.upper()} Model Evaluation:")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Squared Error (MSE): {mse:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        print(f"R-squared (R²): {r2:.6f}")
        
        return results
    
    def get_feature_importance(self, model_name, feature_names=None):
        """Get feature importance for tree-based models"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if model_name == 'random_forest':
            importance = model.feature_importances_
            if feature_names:
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                print(f"\n{model_name.upper()} Feature Importance:")
                print(importance_df)
                
                return importance_df
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def compare_models(self):
        """Compare performance of all trained models"""
        if not self.results:
            print("No models have been evaluated yet.")
            return None
        
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'MAE': results['mae'],
                'MSE': results['mse'],
                'RMSE': results['rmse'],
                'R²': results['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('R²', ascending=False)
        
        print("\nMODEL COMPARISON:")
        print("=" * 60)
        print(comparison_df.to_string(index=False, float_format='%.6f'))
        
        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        best_r2 = comparison_df.iloc[0]['R²']
        
        print(f"\nBest performing model: {best_model} (R² = {best_r2:.6f})")
        
        return comparison_df
    
    def train_all_models(self, X_train, X_test, y_train, y_test, feature_names=None):
        """Train and evaluate all models"""
        print("=" * 60)
        print("TRAINING ALL MODELS")
        print("=" * 60)
        
        # Train OLS model
        print("\n1. Training OLS Model...")
        self.train_ols_model(X_train, y_train, feature_names)
        self.evaluate_model('ols', X_test, y_test)
        
        # Train Linear Regression
        print("\n2. Training Linear Regression...")
        self.train_linear_regression(X_train, y_train)
        self.evaluate_model('linear_regression', X_test, y_test)
        
        # Train Ridge Regression
        print("\n3. Training Ridge Regression...")
        self.train_ridge_regression(X_train, y_train)
        self.evaluate_model('ridge', X_test, y_test)
        
        # Train Random Forest
        print("\n4. Training Random Forest...")
        self.train_random_forest(X_train, y_train)
        self.evaluate_model('random_forest', X_test, y_test)
        
        # Get feature importance for Random Forest
        if feature_names:
            self.get_feature_importance('random_forest', feature_names)
        
        # Compare all models
        print("\n" + "=" * 60)
        comparison = self.compare_models()
        
        return self.models, self.results, comparison