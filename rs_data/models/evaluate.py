from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import numpy as np

def evaluate_model(y_test, y_pred_proba, threshold=0.5):
    """
    Evaluate the model performance and print metrics.

    :param y_test: (array): True labels.
    :param y_pred_proba: (array-like): Predicted probabilities for the positive class.
    :param threshold: (float, optional): Decision threshold for classifying probabilities.
    :return: None
    """
    # Adjust predictions based on threshold
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred_thresh)
    print(f'Accuracy: {accuracy:.2f}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_thresh)
    print('Confusion Matrix:\n', conf_matrix)

    # Classification Report
    class_report = classification_report(y_test, y_pred_thresh)
    print('Classification Report:\n', class_report)

    # ROC-AUC Score
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f'ROC-AUC Score: {roc_auc:.2f}')
    else:
        print("ROC-AUC score is not applicable for multi-class classification.")