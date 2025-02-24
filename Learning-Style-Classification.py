"""
Learning Style Classification Module for Personalized Learning Path Recommendation System

This module contains functions for classifying students into different learning styles
to tailor personalized learning paths.

Techniques Used:
- Clustering
- Classification

Algorithms Used:
- K-Means
- Decision Trees
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class LearningStyleClassification:
    def __init__(self, n_clusters=3):
        """
        Initialize the LearningStyleClassification class.
        
        :param n_clusters: int, number of clusters for K-Means
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.label_encoders = {}  # Store label encoders for each categorical column

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
        Preprocess the data for clustering and classification.
        
        :param data: DataFrame, input data
        :return: DataFrame, preprocessed data
        """
        print("Preprocessing data...")

        # Handle categorical columns
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

    def cluster_students(self, data):
        """
        Cluster students into different learning styles using K-Means.
        
        :param data: DataFrame, input data
        :return: DataFrame, data with cluster labels
        """
        print("Clustering students into learning styles...")
        features = data.drop(columns=['learning_style'], errors='ignore')
        data['learning_style'] = self.kmeans.fit_predict(features)
        print("Clustering completed.")
        return data

    def train_classifier(self, data):
        """
        Train a Decision Tree classifier to predict learning styles.
        
        :param data: DataFrame, input data with cluster labels
        """
        print("Training the Decision Tree classifier...")
        X = data.drop(columns=['learning_style'])
        y = data['learning_style']
        
        # Splitting the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Training the model
        self.decision_tree.fit(X_train, y_train)
        y_pred = self.decision_tree.predict(X_test)
        
        # Evaluating the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print("Decision Tree Classifier Accuracy:", accuracy)
        print("Classification Report:\n", report)

    def save_models(self, output_dir="models"):
        """
        Save the trained models to files.
        
        :param output_dir: str, directory to save models
        """
        print("Saving trained models...")
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.kmeans, os.path.join(output_dir, 'kmeans_model.pkl'))
        joblib.dump(self.decision_tree, os.path.join(output_dir, 'decision_tree_model.pkl'))
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        print(f"Models saved to {output_dir}.")

    def load_models(self, model_dir="models"):
        """
        Load the trained models from files.
        
        :param model_dir: str, directory containing the models
        """
        print("Loading trained models...")
        self.kmeans = joblib.load(os.path.join(model_dir, 'kmeans_model.pkl'))
        self.decision_tree = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
        self.label_encoders = joblib.load(os.path.join(model_dir, 'label_encoders.pkl'))
        print("Models successfully loaded.")

    def classify_new_student(self, student_data):
        """
        Classify a new student into a learning style using the trained Decision Tree classifier.
        
        :param student_data: DataFrame, data of the new student
        :return: int, predicted learning style
        """
        print("Classifying new student...")
        for column in student_data.select_dtypes(include=['object']).columns:
            if column in self.label_encoders:
                student_data[column] = self.label_encoders[column].transform(student_data[column])
        prediction = self.decision_tree.predict(student_data)[0]
        print(f"Predicted learning style: {prediction}")
        return prediction

if __name__ == "__main__":
    # Path to the dataset
    data_filepath = 'data/processed/preprocessed_student_data.csv'

    # Initialize the classifier
    classifier = LearningStyleClassification()

    try:
        # Load and preprocess the data
        data = classifier.load_data(data_filepath)
        data = classifier.preprocess_data(data)

        # Perform clustering and train classifier
        clustered_data = classifier.cluster_students(data)
        classifier.train_classifier(clustered_data)

        # Save models
        classifier.save_models()

    except Exception as e:
        print(f"An error occurred: {e}")
