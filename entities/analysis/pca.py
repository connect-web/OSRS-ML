import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from entities.preprocessing.scaling import stats_scaling
from .plotting import Plot

class Pca(Plot):
    print_variance = False

    def __init__(self, df, n_components=None, **kwargs):
        """
        Initializes the Pca Class

        :param df: The Pandas Dataframe
        :param n_components: The number of components to create the features with. Default is None for Elbow plot
        :param kwargs: The feature group categories that are columns in the dataframe to be scaled as a group.
        """
        super().__init__(df, n_components, **kwargs)
        self.model_name = 'PCA'
        self.df, self.n_components = df, n_components

        feature_groups = [kwargs[key] for key in kwargs if kwargs[key]]  # Filter out None or empty lists
        self.X_scaled = stats_scaling(df, feature_groups=feature_groups)

        self.y = self.df['Banned']

        self.pca = PCA(n_components=self.n_components)

    def run(self):
        """
        Fits the scaled data to the pca
        :return:
        """

        self.X_r = self.pca.fit_transform(self.X_scaled)

        # Explained variance
        if self.print_variance:
            print('explained variance ratio (first two components):', self.pca.explained_variance_ratio_[:2])
            print('sum of explained variance (first two components): ', sum(self.pca.explained_variance_ratio_[:2]))

    def elbow_plot(self):
        """
        Plots an elbow graph to visualize the optimal number of components to capture important features.

        :return: None
        """
        explained_variances = self.pca.explained_variance_ratio_
        cumulative_variances = np.cumsum(explained_variances)

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Different PCA Components')
        plt.grid(True)
        plt.show()