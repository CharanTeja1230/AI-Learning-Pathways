import pandas as pd
import numpy as np

class AdaptiveLearningSystem:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        print("Preprocessing data...")
        if self.data.isnull().values.any():
            self.data.fillna(self.data.mean(numeric_only=True), inplace=True)  # Fill missing values

        print("Dataset columns after preprocessing:", self.data.columns.tolist())
        return self.data

    def train(self, target_column):
        print("Checking dataset before training...")
        print("Columns in dataset:", self.data.columns.tolist())

        if target_column not in self.data.columns:
            raise ValueError(f"Error: Target column '{target_column}' not found in dataset.")

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        print("Training with dataset:", X.shape, "Target:", y.shape)
        # Your training code here...

# Load and train
data_path = 'data/processed/preprocessed_student_data.csv'
data = pd.read_csv(data_path)
learning_system = AdaptiveLearningSystem(data)
processed_data = learning_system.preprocess_data()
learning_system.train('learning_path')
