import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.model_selection import GridSearchCV
import itertools

from rs_data import (PCA, TSNE, UMAP)
from rs_data.database.data import get_bans

from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://localhost:5000")


from ml.datasets.load_set import load_set

from .models import GridModels
from .mlflow_gridsearch import MLflowGridSearchCV

class GridSearchActivity(GridModels):
    def __init__(self, activity : str, classifier_name: str, n_components : int, param_space: dict = None, n_folds: int = 5, new_ban_data: bool = False):
        """
        Loads training data for target activity then completes a grid search on the best available model.

        :param activity: The target activity
        :param classifier_name: The target classifier
        :param n_components: The number of PCA components.
        """
        self.default_parameters = None
        self.activity = activity
        self.classifier_name = classifier_name
        self.n_components = n_components
        self.param_space = param_space
        self.n_folds = n_folds
        self.new_ban_data = new_ban_data

        self.classifier = None

        self.setup_classifier()
        if self.param_space:
            self.get_run_count(self.param_space)

    def setup_classifier(self):
        classifier_func = self.classifier_types.get(self.classifier_name)

        if classifier_func is None:
            raise Exception(f"Classifier {self.classifier_name} is not recognised.")

        param_meta = self.classifier_spaces.get(self.classifier_name)

        if param_meta is None:
            raise Exception(f"Could not find {self.classifier_name} in the classifier_spaces")

        if self.param_space is None:
            # When using grid_search it requires classifier__ prefix for classifier params.
            classifier_space_parameters = {
                f'classifier__{name}' : values
                for name, values in param_meta['search_space'].items()
            }
            self.param_space = classifier_space_parameters


        self.classifier = classifier_func(param_meta['default'])

    def run(self):
        if self.classifier is None:
            raise Exception(f"The classifier was not setup.")

        df , preprocessor = load_set(self.activity, directory='./')

        if self.new_ban_data:
            ban_count = len(df[df['Banned'] == True])
            unbanned_count = len(df[df['Banned'] == False])
            print(f'{ban_count} bans + {unbanned_count} unbans turned into....')
            ban_df = get_bans(df['pid'].tolist())

            mask = df['pid'].isin(ban_df['pid'])
            df.loc[mask, 'Banned'] = ban_df.set_index('pid').loc[df.loc[mask, 'pid'], 'Banned'].values

            ban_count = len(df[df['Banned'] == True])
            unbanned_count = len(df[df['Banned'] == False])
            print(f'{ban_count} bans + {unbanned_count} unbans turned into....')



        X = df.drop(columns=['Banned', 'pid'])
        y = df['Banned']

        mlflow.set_experiment(self.activity)

        pipeline = ImblearnPipeline([
                        ('preprocessor', preprocessor),
                        ('smote', SMOTE(random_state=42)),
                        ('PCA', PCA(n_components=self.n_components)),
                        ('classifier', self.classifier)
                    ])

        grid_search = MLflowGridSearchCV(activity=self.activity,
                                         classifier_name=self.classifier_name,
                                         estimator=pipeline,
                                         param_grid=self.param_space,
                                         cv=5,
                                         verbose=1,
                                         return_train_score=True)

        grid_search.fit(X, y)

    def get_run_count(self, param_grid):
        param_combinations = list(itertools.product(*param_grid.values()))
        n_param_combinations = len(param_combinations)

        # Total number of runs
        n_runs = n_param_combinations * self.n_folds

        print(f"Total number of runs: {n_runs}")
        return n_runs


if __name__ == "__main__":
    GridSearchActivity(
        activity='Dagannoth Prime',
        classifier_name='',
        n_components=0
    )