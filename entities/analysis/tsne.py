from sklearn.manifold import TSNE

from entities.preprocessing.scaling import stats_scaling
from .plotting import Plot

class Tsne(Plot):
    def __init__(self, df, n_components=2, perplexity=30, n_iter=3000, **kwargs):
        """
        Initializes the Tsne class.

        :param df: The pandas DataFrame.
        :param n_components: The number of components to reduce the data to, typically 2 for visualization.
        :param perplexity: The perplexity parameter for t-SNE.
        :param n_iter: The number of iterations for optimization.
        :param **kwargs: Arbitrary keyword arguments representing groups of features to be used in t-SNE.
        """
        super().__init__(df, n_components, **kwargs)
        self.model_name = 't-SNE'
        self.df = df
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.feature_groups = [kwargs[key] for key in kwargs if kwargs[key]]

        self.y = self.df['Banned']  # Assuming 'Banned' is a label column in your DataFrame
        self.X_scaled = stats_scaling(self.df, self.feature_groups)

        self.tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, n_iter=self.n_iter, random_state=42)

    def run(self):
        """
        Runs t-SNE on the scaled data.
        """
        self.X_r = self.tsne.fit_transform(self.X_scaled)