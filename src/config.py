"""
Configuration file for Mobile Phone Price Prediction project
===========================================================

This file contains all configurable parameters for the ML pipeline.
Modify these values to customize the behavior of the analysis.
"""

# Data Configuration
DATA_CONFIG = {
    'dataset_id': 'ganjerlawrence/mobile-phone-price-prediction-cleaned-dataset',
    'raw_data_dir': './data/raw',
    'processed_data_dir': './data/processed',
    'raw_filename': 'mobile_price_data.csv',
    'processed_filename': 'mobile_price_data_cleaned.csv',
    'target_column': 'Price',
    'feature_columns': [
        'Ratings', 'RAM', 'ROM', 'Mobile_Size', 
        'Primary_Cam', 'Selfi_Cam', 'Battery_Power'
    ]
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'apply_log_transform': True,
    'remove_negative_prices': True,
    'scale_features': True
}

# Model Configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    },
    'ridge': {
        'alpha': 1.0,
        'random_state': 42
    },
    'ols': {
        'fit_intercept': True
    }
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'figure_size': {
        'default': (10, 8),
        'comparison': (15, 10),
        'distribution': (15, 5),
        'correlation': (10, 8)
    },
    'style': 'seaborn',
    'color_palette': 'Blues',
    'dpi': 300,
    'save_plots': False,
    'plot_directory': './plots'
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'metrics': [
        'mae',  # Mean Absolute Error
        'mse',  # Mean Squared Error
        'rmse', # Root Mean Squared Error
        'r2',   # R-squared
        'mape'  # Mean Absolute Percentage Error
    ],
    'significance_level': 0.05,
    'performance_thresholds': {
        'excellent': 0.9,
        'very_good': 0.8,
        'good': 0.7,
        'fair': 0.6
    }
}

# Multicollinearity Analysis Configuration
MULTICOLLINEARITY_CONFIG = {
    'vif_threshold_high': 10.0,
    'vif_threshold_moderate': 5.0,
    'correlation_threshold': 0.8
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_models': False,
    'model_directory': './models',
    'save_results': False,
    'results_directory': './results',
    'log_level': 'INFO',
    'verbose': True
}

# Feature Engineering Configuration (for future enhancements)
FEATURE_ENGINEERING_CONFIG = {
    'create_interaction_terms': False,
    'polynomial_features': False,
    'polynomial_degree': 2,
    'feature_selection': False,
    'selection_method': 'recursive'  # 'recursive', 'univariate', 'lasso'
}

# Cross-Validation Configuration (for future enhancements)
CV_CONFIG = {
    'enable_cv': False,
    'cv_folds': 5,
    'cv_random_state': 42,
    'cv_scoring': 'r2'
}

# All configurations combined for easy access
CONFIG = {
    'data': DATA_CONFIG,
    'preprocessing': PREPROCESSING_CONFIG,
    'models': MODEL_CONFIG,
    'visualization': VISUALIZATION_CONFIG,
    'evaluation': EVALUATION_CONFIG,
    'multicollinearity': MULTICOLLINEARITY_CONFIG,
    'output': OUTPUT_CONFIG,
    'feature_engineering': FEATURE_ENGINEERING_CONFIG,
    'cross_validation': CV_CONFIG
}


def get_config(section=None):
    """
    Get configuration parameters
    
    Args:
        section (str, optional): Specific section to retrieve. 
                               If None, returns all configurations.
    
    Returns:
        dict: Configuration parameters
    """
    if section is None:
        return CONFIG
    
    if section in CONFIG:
        return CONFIG[section]
    else:
        raise ValueError(f"Configuration section '{section}' not found. "
                        f"Available sections: {list(CONFIG.keys())}")


def update_config(section, key, value):
    """
    Update a specific configuration parameter
    
    Args:
        section (str): Configuration section
        key (str): Parameter key
        value: New value
    """
    if section not in CONFIG:
        raise ValueError(f"Configuration section '{section}' not found")
    
    CONFIG[section][key] = value
    print(f"Updated {section}.{key} = {value}")


def print_config(section=None):
    """
    Print configuration parameters in a formatted way
    
    Args:
        section (str, optional): Specific section to print
    """
    import json
    
    if section is None:
        config_to_print = CONFIG
        title = "ALL CONFIGURATION PARAMETERS"
    else:
        config_to_print = get_config(section)
        title = f"{section.upper()} CONFIGURATION"
    
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(json.dumps(config_to_print, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    # Demo: Print all configurations
    print_config()
    
    # Demo: Print specific section
    print_config('data')
    
    # Demo: Get specific configuration
    model_params = get_config('models')
    print(f"\nRandom Forest parameters: {model_params['random_forest']}")