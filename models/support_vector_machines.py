import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from .evaluate import evaluate_model  # Ensure this is correctly imported from your project structure

def svm_model(X, y, kernel='linear', C=1.0, probability=True, use_cross_val=False):
    """
    Train and evaluate a Support Vector Machine model.

    :param X: Array-like, shape = [n_samples, n_features], training feature matrix.
    :param y: Array-like, shape = [n_samples], target values.
    :param kernel: String, specifies the kernel type to be used in the algorithm.
    :param C: Float, regularization parameter.
    :param probability: Boolean, whether to enable probability estimates.
    :param use_cross_val: Boolean, whether to use cross-validation for evaluation.
    :return: None
    """
    # Create the SVM classifier
    model = SVC(kernel=kernel, C=C, probability=probability, random_state=42)

    if use_cross_val:
        # Use cross-validation to evaluate model
        y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)  # Convert probabilities to 0 or 1 using a threshold of 0.5
        evaluate_model(y, y_pred, y_pred_proba)
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if probability else None
        evaluate_model(y_test, y_pred, y_pred_proba)