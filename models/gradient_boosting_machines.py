import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from .evaluate import evaluate_model

def gbm_model(X, y, n_estimators=100, learning_rate=0.1, max_depth=3, threshold=0.5, use_cross_val=False):
    """
    Train and evaluate a Gradient Boosting Classifier model.
    """
    print("Training Gradient Boosting Model...")
    gbm = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                     max_depth=max_depth, random_state=42)

    if use_cross_val:
        y_pred_proba = cross_val_predict(gbm, X, y, cv=5, method='predict_proba')[:, 1]
        evaluate_model(y, y_pred_proba, threshold)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        gbm.fit(X_train, y_train)
        y_pred_proba = gbm.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, y_pred_proba, threshold)

import xgboost as xgb

def xgb_model(X, y, n_estimators=100, max_depth=3, learning_rate=0.1, threshold=0.5, use_cross_val=False):
    """
    Train and evaluate an XGBoost model.
    """
    print("Training XGBoost Model...")
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                              random_state=42)

    if use_cross_val:
        y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
        evaluate_model(y, y_pred_proba, threshold)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, y_pred_proba, threshold)

import lightgbm as lgb

def lgbm_model(X, y, n_estimators=100, max_depth=-1, learning_rate=0.1, threshold=0.5, use_cross_val=False):
    """
    Train and evaluate a LightGBM model.
    """
    print("Training LightGBM Model...")
    model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                               random_state=42)

    if use_cross_val:
        y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
        evaluate_model(y, y_pred_proba, threshold)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        evaluate_model(y_test, y_pred_proba, threshold)
