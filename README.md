# kunal
Integrating computational  approaches for the identification  of bioactive compounds in  medicinal mushroom  data analysis project 
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:29:18 2024

@author:kunal
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# File path
FILE_PATH = r"C:\ALY6040_DATAMINING\Assignment2\mushrooms.xlsx"

# Load the data
def load_data(filepath):
    """Load dataset from the specified Excel file."""
    data = pd.read_excel(filepath)
    print("Data loaded successfully.")
    return data

# Preprocess the data
def preprocess_data(data):
    """Preprocess data by encoding categorical variables."""
    data_encoded = pd.get_dummies(data, drop_first=True)
    print("Data preprocessing completed (one-hot encoding applied).")
    return data_encoded

# Split the data into training and testing sets
def split_data(data):
    """Split data into features (X) and target (y), then into training and testing sets."""
    X = data.drop("class_p", axis=1)  # Features
    y = data["class_p"]               # Target (1 for poisonous, 0 for edible)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

# Perform hyperparameter tuning
def tune_decision_tree(X_train, y_train):
    """Tune Decision Tree parameters using GridSearchCV."""
    param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best Decision Tree Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Train ensemble models
def train_ensemble_models(X_train, y_train, base_estimator):
    """Train Random Forest and AdaBoost models with the best Decision Tree as base estimator."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    ada.fit(X_train, y_train)
    
    print("Ensemble models training completed.")
    return rf, ada

# Evaluate the models
def evaluate_model(model, X_test, y_test):
    """Evaluate the model using accuracy score and classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:\n", report)

# Visualize Decision Tree
def plot_decision_tree(model, feature_names):
    """Plot the trained Decision Tree model."""
    plt.figure(figsize=(12, 8))
    tree.plot_tree(model, feature_names=feature_names, class_names=["Edible", "Poisonous"], filled=True)
    plt.show()

# Main function to run the steps
def main():
    # Step 1: Load data
    data = load_data(FILE_PATH)
    
    # Step 2: Preprocess data
    data_encoded = preprocess_data(data)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(data_encoded)
    
    # Step 4: Tune Decision Tree
    best_dt = tune_decision_tree(X_train, y_train)
    
    # Step 5: Train ensemble models
    rf, ada = train_ensemble_models(X_train, y_train, best_dt)
    
    # Step 6: Evaluate models
    print("\nEvaluating Decision Tree:")
    evaluate_model(best_dt, X_test, y_test)
    print("\nEvaluating Random Forest:")
    evaluate_model(rf, X_test, y_test)
    print("\nEvaluating AdaBoost:")
    evaluate_model(ada, X_test, y_test)
    
    # Step 7: Visualize Decision Tree
    print("\nDecision Tree Plot:")
    plot_decision_tree(best_dt, X_train.columns)

# Run the main function
if __name__ == "__main__":
    main()

    
