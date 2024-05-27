import mlflow
import mlflow.sklearn
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, confusion_matrix

scorers = {
    'ROC-AUC': 'roc_auc',
    'accuracy': make_scorer(accuracy_score),
    'recall_class_0': make_scorer(recall_score, average=None, labels=[0]),
    'recall_class_1': make_scorer(recall_score, average=None, labels=[1]),
    'accuracy_class_0': make_scorer(lambda y_true, y_pred: accuracy_score(y_true == 0, y_pred == 0)),
    'accuracy_class_1': make_scorer(lambda y_true, y_pred: accuracy_score(y_true == 1, y_pred == 1))
}

class MLflowGridSearchCV(GridSearchCV):
    METRIC_TYPES = {
        'ROC-AUC': 'mean_test_ROC-AUC',
        'Mean Accuracy': 'mean_test_accuracy',
        'Recall Class 0': 'mean_test_recall_class_0',
        'Recall Class 1': 'mean_test_recall_class_1',
        'Accuracy Class 0': 'mean_test_accuracy_class_0',
        'Accuracy Class 1': 'mean_test_accuracy_class_1',
    }

    def __init__(self, activity, classifier_name,
                 estimator, param_grid, scoring=None, refit='ROC-AUC', **kwargs
                 ):
        if scoring is None:
            scoring = {
                'ROC-AUC': 'roc_auc',
                'accuracy': make_scorer(accuracy_score),
                'recall_class_0': make_scorer(recall_score, average=None, labels=[0]),
                'recall_class_1': make_scorer(recall_score, average=None, labels=[1]),
                'accuracy_class_0': make_scorer(lambda y_true, y_pred: accuracy_score(y_true == 0, y_pred == 0)),
                'accuracy_class_1': make_scorer(lambda y_true, y_pred: accuracy_score(y_true == 1, y_pred == 1))
            }

        print(f'Param grid: {param_grid}')
        super().__init__(estimator, param_grid, scoring=scoring, refit=refit, n_jobs=12, **kwargs)

        self.activity = activity
        self.classifier_name = classifier_name
        mlflow.set_experiment(self.activity)

    def get_mlflow_params(self):
        return {
            "PCA_n_components": self.estimator.named_steps['PCA'].n_components,
            "Sampling": "SMOTE",
            "Classifier": self.classifier_name,
        }

    def log_all_results(self, X, y):

        for i in range(len(self.cv_results_['mean_test_accuracy'])):
            with mlflow.start_run(nested=True):
                # add tag to run
                run_tag = 'best_grid_search' if i == self.best_index_ else 'grid_search'
                mlflow.set_tags({run_tag: True})

                # Get default params & grid search params
                default_params = self.get_mlflow_params()
                grid_params = {key: value for key,value in self.cv_results_['params'][i].items()}

                default_params.update(grid_params)

                # get the metric results
                metric_results = {
                    metric_type: self.cv_results_[grid_metric_type][i]
                    for metric_type, grid_metric_type in self.METRIC_TYPES.items()
                }

                # log params + metrics as batch
                mlflow.log_params(default_params)
                mlflow.log_metrics(metric_results)

                # Refit the estimator on the entire dataset with the current params
                if i == self.best_index_:
                    pipeline = clone(self.estimator)
                    pipeline.set_params(**grid_params)
                    pipeline.fit(X, y)

                    # Log the model
                    mlflow.sklearn.log_model(pipeline, "model")



    def fit(self, X, y=None, **fit_params):
        # Start the MLflow experiment
        with mlflow.start_run():
            # Log initial parameters
            mlflow.log_param("PCA_n_components", self.estimator.named_steps['PCA'].n_components)
            mlflow.log_param("Sampling", "SMOTE")
            mlflow.log_param("Classifier", self.classifier_name)

            # Perform grid search
            super().fit(X, y, **fit_params)

            # Log all parameter sets and corresponding metrics
            self.log_all_results(X, y)

            print(self.best_score_)

