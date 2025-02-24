import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessing:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def load_data(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: File not found at {filepath}")
        
        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded successfully from {filepath}")
            return data
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

    def clean_data(self, data):
        if data is None or data.empty:
            raise ValueError("No data to clean.")
        
        data = data.drop_duplicates()
        print(f"Removed duplicates. New shape: {data.shape}")
        
        numeric_data = data.select_dtypes(include=[np.number])
        data[numeric_data.columns] = self.imputer.fit_transform(numeric_data)
        print("Missing values imputed.")
        
        return data

    def normalize_data(self, data, columns):
        if data is None or data.empty:
            raise ValueError("No data to normalize.")

        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        data[columns] = self.scaler.fit_transform(data[columns])
        print("Data normalized.")
        
        return data

    def extract_features(self, data):
        # Detect dataset type based on column names
        if all(col in data.columns for col in ['math_score', 'reading_score', 'writing_score']):
            required_columns = ['math_score', 'reading_score', 'writing_score']
            data['avg_performance'] = data[required_columns].mean(axis=1)
            print("Feature extraction for student data complete.")
        elif all(col in data.columns for col in ['Progress (%)', 'Completion_Time (hrs)', 'Engagement_Score']):
            required_columns = ['Progress (%)', 'Completion_Time (hrs)', 'Engagement_Score']
            data['avg_learning_progress'] = data[['Progress (%)', 'Engagement_Score']].mean(axis=1)
            print("Feature extraction for adaptive learning data complete.")
        else:
            print("Skipping feature extraction: No matching column patterns found.")
        
        return data

    def preprocess(self, filepath, columns_to_normalize):
        data = self.load_data(filepath)
        data = self.clean_data(data)
        data = self.extract_features(data)
        
        # Only normalize columns that exist in the dataset
        valid_columns = [col for col in columns_to_normalize if col in data.columns]
        if valid_columns:
            data = self.normalize_data(data, valid_columns)
        else:
            print("Skipping normalization: No valid columns found in dataset.")
        
        print("Preprocessing complete.")
        return data

if __name__ == "__main__":
    student_data_filepath = 'data/raw/student_data.csv'
    adaptive_learning_filepath = 'data/raw/adaptive_learning_dataset.csv'
    output_filepath_student = 'data/processed/preprocessed_student_data.csv'
    output_filepath_adaptive = 'data/processed/pre_processed_data/adaptive_learning_dataset.csv'
    os.makedirs(os.path.dirname(output_filepath_student), exist_ok=True)
    os.makedirs(os.path.dirname(output_filepath_adaptive), exist_ok=True)
    
    preprocessing = DataPreprocessing()
    
    try:
        # Process student data
        columns_to_normalize_student = ['math_score', 'reading_score', 'writing_score', 'avg_performance']
        preprocessed_student_data = preprocessing.preprocess(student_data_filepath, columns_to_normalize_student)
        preprocessed_student_data.to_csv(output_filepath_student, index=False)
        print(f"Preprocessed student data saved to {output_filepath_student}")
        
        # Process adaptive learning dataset
        columns_to_normalize_adaptive = ['Progress (%)', 'Completion_Time (hrs)', 'Engagement_Score', 'avg_learning_progress']
        preprocessed_adaptive_data = preprocessing.preprocess(adaptive_learning_filepath, columns_to_normalize_adaptive)
        preprocessed_adaptive_data.to_csv(output_filepath_adaptive, index=False)
        print(f"Preprocessed adaptive learning data saved to {output_filepath_adaptive}")
    except Exception as e:
        print(f"An error occurred: {e}")
