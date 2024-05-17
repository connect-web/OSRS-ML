from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from itertools import product
from sklearn.model_selection import cross_val_score
import numpy as np

preprocessing_options = {
    'PCA': PCA(n_components=2),
    't-SNE': TSNE(n_components=2),
    'UMAP': UMAP(n_components=2)
}

smote_options = {
    'regular': SMOTE(random_state=42),
    'borderline': BorderlineSMOTE(random_state=42),
    'svm': SVMSMOTE(random_state=42),
    #'kmeans': KMeansSMOTE(random_state=42)
}

classifiers = {
    'Random Forest': RandomForestClassifier(),
    #'GBM': GradientBoostingClassifier(),
    #'Logistic Regression': [LogisticRegression(solver=s) for s in ['liblinear', 'saga']],
    #'SVM': [SVC(kernel=k) for k in ['linear', 'rbf', 'poly', 'sigmoid']],
    #'LGBM': LGBMClassifier(),
    #'XGBoost': XGBClassifier(),
    #'Extra Trees': ExtraTreesClassifier()
}

def create_pipelines(preprocessing_options, smote_options, classifiers):
    for preproc_key, smote_key, classifier_key in product(preprocessing_options, smote_options, classifiers):
        classifier_list = classifiers[classifier_key]
        if not isinstance(classifier_list, list):
            classifier_list = [classifier_list]  # Ensure it is iterable
        for classifier in classifier_list:
            pipeline = ImblearnPipeline([
                ('preprocessing', preprocessing_options[preproc_key]),
                ('smote', smote_options[smote_key]),
                ('classifier', classifier)
            ])
            config_name = f"{preproc_key}-{smote_key}-{classifier_key}-{classifier.get_params()}"
            yield pipeline, config_name
def evaluate_pipelines(pipelines, X, y, show_results=False):
    results = {}
    for pipeline, config_name in pipelines:
        scores = cross_val_score(pipeline, X, y, cv=5)  # Change cv parameter as needed
        if show_results:
            print(f"Score: {scores[0]:.3f} ± {scores[1]:.3f}")
        results[config_name] = np.mean(scores), np.std(scores)
    return results

pipeline_configs = list(create_pipelines(preprocessing_options, smote_options, classifiers))

def example_usage(X,y):
    pipeline_results = evaluate_pipelines(pipeline_configs, X, y)
    for config, scores in pipeline_results.items():
        print(f"Configuration: {config} - Score: {scores[0]:.3f} ± {scores[1]:.3f}")