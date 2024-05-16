from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
from .evaluate import evaluate_model


def rf_model(X, y, class_weights=None, n_estimators=50, threshold=0.5, use_cross_val=False):
    """
    Train and evaluate a RandomForestClassifier model.

    :param X: (array-like) The training features.
    :param y: (array-like) True labels.
    :param class_weights: (dict, optional) A dictionary of the class weights.
    :param n_estimators: (int, optional) Number of estimators in the forest.
    :param threshold: (float, optional) Decision threshold for classifying probabilities.
    :param use_cross_val: (bool, optional) Whether to use cross-validation for evaluation.
    :return: None
    """
    print(f'Weights: {class_weights}')

    # Create the Random Forest classifier
    if class_weights is None:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight=class_weights)

    if use_cross_val:
        # Use cross-validation for evaluation
        y_pred_proba = cross_val_predict(rf, X, y, cv=5, method='predict_proba')[:, 1]
        evaluate_model(y, y_pred_proba, threshold)
    else:
        # Split the data into training and testing sets for demonstrating these metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf.fit(X_train, y_train)

        # Predictions
        y_pred_proba = rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        evaluate_model(y_test, y_pred_proba, threshold)


def et_model(X, y, class_weights=None, n_estimators=50, threshold=0.5, use_cross_val=False):
    """
    Train and evaluate an ExtraTreesClassifier model.

    :param X: (array-like) The training features.
    :param y: (array-like) True labels.
    :param class_weights: (dict, optional) A dictionary of the class weights.
    :param n_estimators: (int, optional) Number of estimators in the forest.
    :param threshold: (float, optional) Decision threshold for classifying probabilities.
    :param use_cross_val: (bool, optional) Whether to use cross-validation for evaluation.
    :return: None
    """
    print(f'Weights: {class_weights}')

    # Create the Extra Trees classifier
    if class_weights is None:
        et = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42)
    else:
        et = ExtraTreesClassifier(n_estimators=n_estimators, random_state=42, class_weight=class_weights)

    if use_cross_val:
        # Use cross-validation for evaluation
        y_pred_proba = cross_val_predict(et, X, y, cv=5, method='predict_proba')[:, 1]
        evaluate_model(y, y_pred_proba, threshold)
    else:
        # Split the data into training and testing sets for demonstrating these metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        et.fit(X_train, y_train)

        # Predictions
        y_pred_proba = et.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        evaluate_model(y_test, y_pred_proba, threshold)