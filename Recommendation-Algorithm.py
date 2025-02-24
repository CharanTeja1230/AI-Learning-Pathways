"""
Recommendation Algorithm Module for Personalized Learning Path Recommendation System

This module contains functions for developing a recommendation algorithm to suggest personalized
learning paths for students based on their learning styles and progress.

Techniques Used:
- Collaborative Filtering
- Content-Based Filtering
- Hybrid Approach

Libraries/Tools:
- TensorFlow
- PyTorch
- scikit-learn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os

class RecommendationAlgorithm:
    def __init__(self):
        """
        Initialize the RecommendationAlgorithm class.
        """
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.label_encoders = {}

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        try:
            data = pd.read_csv(filepath)
            print("Data successfully loaded.")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {filepath} was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading data: {e}")

    def preprocess_data(self, data):
        """
        Preprocess the data for recommendation.
        
        :param data: DataFrame, input data
        :return: DataFrame, preprocessed data
        """
        print("Preprocessing data...")

        # Encode categorical columns
        for column in data.select_dtypes(include=['object']).columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                data[column] = self.label_encoders[column].fit_transform(data[column])
            else:
                data[column] = self.label_encoders[column].transform(data[column])

        # Fill missing numeric values
        data = data.fillna(data.mean(numeric_only=True))

        print("Missing values handled and categorical data encoded.")
        return data

    def fit(self, data):
        """
        Fit the recommendation model.
        
        :param data: DataFrame, input data
        """
        print("Fitting the recommendation model...")
        features = data.drop(columns=['student_id'], errors='ignore')
        self.model.fit(features)
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/recommendation_model.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        print("Model training completed and saved.")

    def recommend_learning_paths(self, student_data, n_recommendations=5):
        """
        Recommend learning paths for a student based on their data.
        
        :param student_data: DataFrame, data of the student
        :param n_recommendations: int, number of recommendations to provide
        :return: list, recommended learning paths
        """
        print("Loading the recommendation model...")
        self.model = joblib.load('models/recommendation_model.pkl')
        self.label_encoders = joblib.load('models/label_encoders.pkl')

        # Preprocess the student data
        for column in student_data.select_dtypes(include=['object']).columns:
            if column in self.label_encoders:
                student_data[column] = self.label_encoders[column].transform(student_data[column])

        distances, indices = self.model.kneighbors(student_data, n_neighbors=n_recommendations)
        return indices.flatten().tolist()

if __name__ == "__main__":
    data_filepath = 'data/processed/preprocessed_student_data.csv'
    student_data_filepath = 'data/processed/new_student_data.csv'

    recommender = RecommendationAlgorithm()

    try:
        # Load and preprocess the data
        data = recommender.load_data(data_filepath)
        data = recommender.preprocess_data(data)

        # Fit the recommendation model
        recommender.fit(data)

        # Load and preprocess new student data
        new_student_data = recommender.load_data(student_data_filepath)
        new_student_data = recommender.preprocess_data(new_student_data)

        # Recommend learning paths
        recommended_paths = recommender.recommend_learning_paths(new_student_data)
        print("Recommended Learning Paths for the Student:", recommended_paths)

    except Exception as e:
        print(f"An error occurred: {e}")
