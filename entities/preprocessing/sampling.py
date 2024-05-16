import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


class DataBalancer:
    smote_options = {
        'regular': SMOTE(random_state=42),
        'borderline': BorderlineSMOTE(random_state=42),
        'svm': SVMSMOTE(random_state=42),
        'kmeans': KMeansSMOTE(random_state=42)
    }
    def __init__(self, df, target_column='Banned'):
        self.df = df
        self.target_column = target_column
        self.features = df.drop(columns=[target_column])
        self.labels = df[target_column]
        self.model = RandomForestClassifier(random_state=42)

    def apply_smote(self, variant='regular'):
        """
        Applies SMOTE to oversample the minority class and constructs the DataFrame in one go to avoid fragmentation.

        :param variant: The type of SMOTE variant to apply. Options are ['regular', 'borderline', 'svm', 'kmeans'].
        :return: The balanced DataFrame.
        """

        smote = self.smote_options.get(variant, SMOTE(random_state=42))
        X_res, y_res = smote.fit_resample(self.features, self.labels)

        # Ensure all values are non-negative
        X_res = np.where(X_res < 0, 0, X_res)

        self.balanced_df = pd.DataFrame(np.c_[X_res, y_res], columns=list(self.features.columns) + [self.target_column])
        print(f"Resampling complete using {variant} variant. Balanced data size: {len(self.balanced_df)} rows.")


        return self.balanced_df

    def perform_cross_validation(self, cv=5):
        """
        Performs cross-validation on the dataset and returns classification reports for each fold.

        :param cv: number of cross-validation folds to use.
        :return: a list of tuples containing the cross-validation predictions coupled with the real target values.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        results = []

        for train_index, test_index in skf.split(self.df.drop(columns=[self.target_column]),
                                                 self.df[self.target_column]):
            X_train, X_test = self.df.iloc[train_index].drop(columns=[self.target_column]), self.df.iloc[
                test_index].drop(columns=[self.target_column])
            y_train, y_test = self.df.iloc[train_index][self.target_column], self.df.iloc[test_index][
                self.target_column]

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)
            results.append((y_test, predictions))

        return results

    def evaluate_cross_validation(self, results):
        """
        Evaluates the cross-validation results and prints the classification report and confusion matrix for each fold.
        """
        for i, (y_test, predictions) in enumerate(results, 1):
            print(f"Results for fold {i}")
            print(classification_report(y_test, predictions))
            cm = confusion_matrix(y_test, predictions)
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title('Confusion Matrix')
            plt.ylabel('Actual Labels')
            plt.xlabel('Predicted Labels')
            plt.show()

    def visualize_balancing_effect(self):
        """
        Visualize the effect of balancing by comparing label distributions before and after.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(x=self.labels, ax=ax[0])
        ax[0].set_title('Original Data Distribution')
        ax[0].set_xlabel('Class')
        ax[0].set_ylabel('Frequency')

        sns.countplot(x=self.df[self.target_column], ax=ax[1])
        ax[1].set_title('Balanced Data Distribution')
        ax[1].set_xlabel('Class')
        ax[1].set_ylabel('Frequency')

        plt.show()


def undersample(df):
    """
    Undersamples the majority class in a dataframe based on the specified limit.

    This is a poor performance method to undersample the majority class randomly.

    :param df: Dataframe to undersample
    :return: Dataframe with majority class under sampled.
    """
    # Split the DataFrame into majority and minority classes
    df_a = df[df['Banned'] == 0]
    df_b = df[df['Banned'] == 1]

    limit = min([len(df_a),len(df_b)])

    if len(df_a) < len(df_b):
        majority_df = df_b
        minority_df = df_a
    else:
        majority_df = df_a
        minority_df = df_b

    # Check if the majority class needs to be undersampled
    if len(majority_df) > limit:
        # Randomly sample from the majority dataframe without replacement
        majority_df = majority_df.sample(n=limit, random_state=42)



    # Concatenate the minority and the undersampled majority dataframes
    undersampled_df = pd.concat([majority_df, minority_df], ignore_index=True)

    print(f'Undersampling complete: {len(df_a)+len(df_b)} into {len(undersampled_df)} rows.')
    print(f'Class size: {limit}')
    return undersampled_df