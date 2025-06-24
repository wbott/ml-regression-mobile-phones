#!/usr/bin/env python3
"""
Mobile Phone Price Prediction - Main Pipeline
==============================================

This script orchestrates the complete machine learning pipeline for 
mobile phone price prediction, including data preprocessing, exploratory 
data analysis, model training, and evaluation.

Usage:
    python main.py [--skip-download] [--skip-eda] [--models MODEL1,MODEL2,...]

Author: ML Regression Mobile Phones Project
"""

import warnings
import argparse
import sys
import os

# Add src to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data_preprocessing import DataPreprocessor
from eda import ExploratoryDataAnalysis
from models import ModelTrainer
from evaluation import ModelEvaluator

warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mobile Phone Price Prediction Pipeline")
    
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip dataset download (use existing data)')
    parser.add_argument('--skip-eda', action='store_true',
                       help='Skip exploratory data analysis')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of models to train (ols,ridge,random_forest,all)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    return parser.parse_args()


def main():
    """Main pipeline execution"""
    print("="*70)
    print("MOBILE PHONE PRICE PREDICTION - ML PIPELINE")
    print("="*70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Step 1: Data Preprocessing
    print("\n" + "="*50)
    print("STEP 1: DATA PREPROCESSING")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    
    if not args.skip_download:
        print("Downloading dataset...")
        preprocessor.download_dataset()
    else:
        print("Skipping dataset download...")
    
    print("Loading and preprocessing data...")
    data = preprocessor.preprocess_pipeline()
    
    X_train = data['X_train']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    df_cleaned = data['df_cleaned']
    
    print(f"✅ Data preprocessing completed!")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Test samples: {len(X_test)}")
    print(f"   - Features: {len(feature_names)}")
    
    # Step 2: Exploratory Data Analysis
    if not args.skip_eda:
        print("\n" + "="*50)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        eda = ExploratoryDataAnalysis(df_cleaned)
        eda_results = eda.comprehensive_eda()
        
        print("✅ EDA completed!")
    else:
        print("\n⏭️  Skipping EDA...")
    
    # Step 3: Model Training and Evaluation
    print("\n" + "="*50)
    print("STEP 3: MODEL TRAINING & EVALUATION")
    print("="*50)
    
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Determine which models to train
    if args.models.lower() == 'all':
        models_to_train = ['ols', 'ridge', 'random_forest']
    else:
        models_to_train = [m.strip().lower() for m in args.models.split(',')]
    
    print(f"Training models: {', '.join(models_to_train)}")
    
    # Train and evaluate each model
    model_results = {}
    
    for model_name in models_to_train:
        print(f"\n🔄 Training {model_name.upper()} model...")
        
        try:
            if model_name == 'ols':
                model = trainer.train_ols_model(X_train_scaled, y_train, feature_names)
                predictions = trainer.make_predictions('ols', X_test_scaled)
                
            elif model_name == 'ridge':
                model = trainer.train_ridge_regression(X_train_scaled, y_train)
                predictions = trainer.make_predictions('ridge', X_test_scaled)
                
            elif model_name == 'random_forest':
                model = trainer.train_random_forest(X_train, y_train)  # Use unscaled data for RF
                predictions = trainer.make_predictions('random_forest', X_test)
                
            else:
                print(f"❌ Unknown model: {model_name}")
                continue
            
            # Evaluate model
            metrics, residuals = evaluator.comprehensive_evaluation(y_test, predictions, model_name)
            model_results[model_name] = {
                'model': model,
                'predictions': predictions,
                'metrics': metrics,
                'y_true': y_test,
                'y_pred': predictions
            }
            
            print(f"✅ {model_name.upper()} model completed!")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
    
    # Step 4: Model Comparison
    print("\n" + "="*50)
    print("STEP 4: MODEL COMPARISON")
    print("="*50)
    
    if len(model_results) > 1:
        # Compare models visually
        comparison_data = {name: {'y_true': data['y_true'], 'y_pred': data['y_pred']} 
                          for name, data in model_results.items()}
        evaluator.compare_models_visualization(comparison_data)
    
    # Create comprehensive comparison table
    comparison_table = evaluator.create_metrics_comparison_table()
    
    # Generate performance insights
    best_model, worst_model = evaluator.model_performance_summary()
    
    # Step 5: Feature Importance (for Random Forest)
    if 'random_forest' in model_results:
        print("\n" + "="*50)
        print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        trainer.get_feature_importance('random_forest', feature_names)
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED")
    print("="*70)
    print(f"✅ Successfully trained {len(model_results)} models")
    print(f"🏆 Best performing model: {best_model}")
    print(f"📊 Results saved in model objects and evaluator")
    print("\nTo access results:")
    print("- Model objects: trainer.models")
    print("- Evaluation metrics: evaluator.evaluation_results")
    print("- Comparison table: comparison_table")
    
    return {
        'preprocessor': preprocessor,
        'trainer': trainer,
        'evaluator': evaluator,
        'model_results': model_results,
        'comparison_table': comparison_table,
        'best_model': best_model
    }


if __name__ == "__main__":
    results = main()