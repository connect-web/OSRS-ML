from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

class GridModels:
    classifier_types = {
        "RandomForest": lambda x: RandomForestClassifier(**x),
        "ExtraTrees": lambda x: ExtraTreesClassifier(**x),
        "GradientBoosting": lambda x: GradientBoostingClassifier(**x),
        "SVM": lambda x: SVC(**x),
        "LogisticRegression": lambda x: LogisticRegression(**x),
        "LGBMClassifier": lambda x: LGBMClassifier(**x),
        "XGBClassifier": lambda x: XGBClassifier(**x)
    }

    classifier_spaces = {
        "RandomForest": {
            'default': {'n_estimators': 100, 'random_state': 42, 'n_jobs': 12},
            'search_space': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20],
                             'min_samples_split': [2, 5, 10]}
        },
        "ExtraTrees": {
            'default': {'n_estimators': 100, 'random_state': 42, 'n_jobs': 12},
            'search_space': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20],
                             'min_samples_leaf': [1, 2, 4]}
        },
        "GradientBoosting": {
            'default': {'random_state': 42},
            'search_space': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2],
                             'max_depth': [3, 5, 10]}
        },
        "SVM": {
            'default': {'probability': True, 'random_state': 42},
            'search_space': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        },
        "LogisticRegression": {
            'default': {'random_state': 42, 'n_jobs': 12, 'max_iter': 1000},
            'search_space': {'C': [0.1,0.5, 1, 5, 10], 'solver': ['liblinear', 'saga']}
        },
        "LGBMClassifier": {
            'default': {'random_state': 42, 'n_jobs': 12},
            'search_space': {'num_leaves': [31, 50, 100], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [-1, 10, 20]}
        },
        "XGBClassifier": {
            'default': {'use_label_encoder': False, 'random_state': 42, 'n_jobs': 12, 'eval_metric': 'logloss'},
            'search_space': {'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 10], 'min_child_weight': [1, 5, 10]}
        }
    }