import os
import numpy as np
import pandas as pd
import kaggle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, dataset_id="ganjerlawrence/mobile-phone-price-prediction-cleaned-dataset"):
        self.dataset_id = dataset_id
        self.raw_data_dir = "./data/raw"
        self.processed_data_dir = "./data/processed"
        self.raw_filename = "mobile_price_data.csv"
        self.processed_filename = "mobile_price_data_cleaned.csv"
        self.scaler = StandardScaler()
        
    def download_dataset(self):
        """Download dataset from Kaggle"""
        os.makedirs(self.raw_data_dir, exist_ok=True)
        kaggle.api.dataset_download_files(self.dataset_id, path=self.raw_data_dir, unzip=True)
        
        # Rename the downloaded file to our standard naming
        original_file = os.path.join(self.raw_data_dir, "Mobile-Price-Prediction-cleaned_data.csv")
        renamed_file = os.path.join(self.raw_data_dir, self.raw_filename)
        if os.path.exists(original_file) and not os.path.exists(renamed_file):
            os.rename(original_file, renamed_file)
        
        print(f"Dataset downloaded and extracted to {self.raw_data_dir}")
        
    def load_raw_data(self):
        """Load raw dataset from CSV file"""
        file_path = os.path.join(self.raw_data_dir, self.raw_filename)
        if not os.path.exists(file_path):
            self.download_dataset()
        
        df = pd.read_csv(file_path)
        return df
    
    def save_processed_data(self, df):
        """Save processed data to CSV file"""
        os.makedirs(self.processed_data_dir, exist_ok=True)
        file_path = os.path.join(self.processed_data_dir, self.processed_filename)
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
    
    def load_processed_data(self):
        """Load processed dataset if it exists"""
        file_path = os.path.join(self.processed_data_dir, self.processed_filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Loaded processed data from {file_path}")
            return df
        else:
            print("No processed data found. Run preprocessing first.")
            return None
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        target_col = "Price"
        
        # Remove rows with negative prices
        df_cleaned = df[df[target_col] >= 0].copy()
        print(f"Removed {len(df) - len(df_cleaned)} rows with negative prices")
        
        # Apply log transformation to stabilize variance
        df_cleaned["LogPrice"] = np.log1p(df_cleaned[target_col])
        
        # Drop original Price column and rename LogPrice
        df_cleaned.drop(columns=[target_col], inplace=True)
        df_cleaned.rename(columns={"LogPrice": "Price"}, inplace=True)
        
        return df_cleaned
    
    def get_features_and_target(self, df):
        """Split data into features and target"""
        features = ['Ratings', 'RAM', 'ROM', 'Mobile_Size', 'Primary_Cam', 'Selfi_Cam', 'Battery_Power']
        X = df[features]
        y = df['Price']
        return X, y, features
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, use_cached=True):
        """Complete preprocessing pipeline"""
        # Try to load processed data if it exists and use_cached is True
        if use_cached:
            df_cleaned = self.load_processed_data()
            if df_cleaned is not None:
                print("Using cached processed data")
            else:
                df_cleaned = None
        else:
            df_cleaned = None
        
        # If no cached data, process from raw
        if df_cleaned is None:
            print("Processing raw data...")
            # Load and clean data
            df_raw = self.load_raw_data()
            df_cleaned = self.clean_data(df_raw)
            
            # Save processed data
            self.save_processed_data(df_cleaned)
        
        # Get features and target
        X, y, feature_names = self.get_features_and_target(df_cleaned)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'df_cleaned': df_cleaned,
            'df_raw': df_raw if 'df_raw' in locals() else None
        }