import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImblearnPipeline
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from rs_data import get_dataframe, SkillType
from rs_data import (PCA, TSNE, UMAP)
from rs_data.database.rs_processing import Leaderboards


class SkillExperiments:
    user_limit = 1000
    skill_type = SkillType.EXPERIENCE
    def __init__(self, N_JOBS = 16):
        self.completed_acitivties = []
        activities = Leaderboards.get_skill_names(keep_overall=True)
        self.activities = [activity for activity in activities if activity not in self.completed_acitivties]
        print(f'You have {len(self.activities)} activities to complete!')

        # Configure MLflow to connect to your local server
        mlflow.set_tracking_uri("http://localhost:5000")
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.pca_components = [
            20, 30
        ]
        self.classifiers = [
            ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)),
            ("ExtraTrees", ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)),
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),  # Does not support n_jobs
            ("SVM", SVC(probability=True, random_state=42)),  # Does not support n_jobs
            ("LogisticRegression", LogisticRegression(random_state=42, n_jobs=N_JOBS)),
            ("LGBMClassifier", LGBMClassifier(random_state=42, n_jobs=N_JOBS)),
            ("XGBClassifier",
             XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=N_JOBS))
        ]
    def get_name(self, activity):
        return f"2) {activity} {self.skill_type.description} Model comparison for {self.user_limit} users"
    def create_experiments(self):
        for activity in self.activities:
            try:
                mlflow.set_experiment(self.get_name(activity))  # Set your experiment name
            except:
                print(f"Created {activity} experiment")

    def get_data(self, activity):
        df, formatter = get_dataframe(activity, limit=self.user_limit, aggregate=True, skill_type=self.skill_type, extra_features=False)

        # Creating the ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ('std', StandardScaler(), formatter.agg_skills + formatter.agg_minigames),
            ('minmax', MinMaxScaler(), formatter.live_skills),
            ('minmax_2', MinMaxScaler(), formatter.live_minigames),
        ])

        X = df.drop(columns=['Banned', 'pid'])
        y = df['Banned']
        return X, y, preprocessor

    def run(self):
        self.create_experiments()

        for activity in self.activities:
            self.run_activity(activity)

    def run_activity(self, activity):
        mlflow.set_experiment(self.get_name(activity))

        X, y, preprocessor = self.get_data(activity)

        for pca_n_components in self.pca_components:
            for name, classifier in self.classifiers:
                with mlflow.start_run():
                    # Create the pipeline
                    pipeline = ImblearnPipeline([
                        ('preprocessor', preprocessor),
                        ('smote', SMOTE(random_state=42)),
                        ('PCA', PCA(n_components=pca_n_components)),
                        ('classifier', classifier)
                    ])

                    # Log pipeline components and PCA components
                    mlflow.log_param("PCA_n_components", pca_n_components)
                    mlflow.log_param("Classifier", name)
                    mlflow.log_param("Sampling", "SMOTE")

                    # Calculate scores
                    accuracy_scores = cross_val_score(pipeline, X, y, cv=self.cv, scoring='accuracy')
                    y_pred_proba = cross_val_predict(pipeline, X, y, cv=self.cv)

                    y_pred = (y_pred_proba >= 0.5).astype(int)
                    accuracy_per_class = [
                        accuracy_score(y == k, y_pred == k) for k in [0, 1]
                    ]
                    roc_auc = roc_auc_score(y, y_pred)

                    recall_per_class = recall_score(y, y_pred, average=None)

                    # Confusion matrix
                    conf_matrix = confusion_matrix(y, y_pred)

                    mlflow.log_metric("Mean Accuracy", np.mean(accuracy_scores))
                    mlflow.log_metric("ROC-AUC", roc_auc)
                    mlflow.log_metric("Recall Class 0", recall_per_class[0])
                    mlflow.log_metric("Recall Class 1", recall_per_class[1])

                    mlflow.log_metric("Accuracy Class 0", accuracy_per_class[0])
                    mlflow.log_metric("Accuracy Class 1", accuracy_per_class[1])

                    df_conf_matrix = pd.DataFrame(conf_matrix, index=["True Neg", "True Pos"],
                                                  columns=["Pred Neg", "Pred Pos"])
                    conf_matrix_file_path = f"confusion_matrix_{name}_pca{pca_n_components}.csv"
                    df_conf_matrix.to_csv(conf_matrix_file_path)
                    mlflow.log_artifact(conf_matrix_file_path)
                    os.remove(conf_matrix_file_path)

                    mlflow.sklearn.log_model(pipeline, "model")

                    mlflow.end_run()


if __name__ == '__main__':
    SkillExperiments().run()
