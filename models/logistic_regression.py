import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from .evaluate import evaluate_model

def logistic_regression_model(X, y, C=1.0, penalty='l2', solver='liblinear', use_cross_val=False):
    """
    Train and evaluate a logistic regression model.

    :param X: Array-like, feature matrix.
    :param y: Array-like, labels.
    :param C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    :param penalty: Used to specify the norm used in the penalization.
    :param solver: Algorithm to use in the optimization problem.
    :param use_cross_val: Whether to use cross-validation for evaluation.
    """
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, random_state=42)

    if use_cross_val:
        # Perform cross-validation and use probabilities to evaluate ROC AUC
        y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)  # Convert probabilities to 0 or 1 using a threshold of 0.5
        evaluate_model(y, y_pred, y_pred_proba)
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Making predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        evaluate_model(y_test, y_pred, y_pred_proba)
