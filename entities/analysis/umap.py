from umap import UMAP  # Make sure UMAP is properly installed and imported

from entities.preprocessing.scaling import stats_scaling
from .plotting import Plot  # Assuming you have a base class Plot that handles common plotting functions

class Umap(Plot):
    def __init__(self, df, n_components=2, n_neighbors=15, min_dist=0.1, **kwargs):
        """
        Initializes the Umap class.

        :param df: The pandas DataFrame.
        :param n_components: The number of components to reduce the data to, typically 2 for visualization.
        :param n_neighbors: The number of neighboring points used in manifold approximation.
        :param min_dist: The effective minimum distance between embedded points.
        :param **kwargs: Arbitrary keyword arguments representing groups of features to be used in UMAP.
        """
        super().__init__(df, n_components, **kwargs)
        self.model_name = 'UMAP'
        self.df = df
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.feature_groups = [kwargs[key] for key in kwargs if kwargs[key]]

        self.y = self.df['Banned']  # Assuming 'Banned' is a label column in your DataFrame
        self.X_scaled = stats_scaling(self.df, self.feature_groups)

        self.umap = UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors,
                         min_dist=self.min_dist, random_state=42)

    def run(self):
        """
        Runs UMAP on the scaled data.
        """
        self.X_r = self.umap.fit_transform(self.X_scaled)
