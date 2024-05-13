from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import numpy as np

def rf_model(X, y, class_weights, n_estimators=50, threshold = 0.5):

    print(f'Weights: {class_weights}')

    # Split the data into training and testing sets for demonstrating these metrics
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    # Create and train the Random Forest classifier
    if class_weights is None:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight=class_weights)

    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    # Adjust threshold
    y_pred_proba = (y_pred_proba >= threshold).astype(int)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', conf_matrix)

    # Classification Report
    class_report = classification_report(y_test, y_pred)
    print('Classification Report:\n', class_report)

    # ROC-AUC Score
    # Note: ROC-AUC can only be calculated for binary classification tasks
    if len(np.unique(y)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f'ROC-AUC Score: {roc_auc:.2f}')
    else:
        print("ROC-AUC score is not applicable for multi-class classification.")
