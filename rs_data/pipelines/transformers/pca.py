from .base_transformer import BaseTransformation
from sklearn.decomposition import PCA as SklearnPCA
import numpy as np
import matplotlib.pyplot as plt

class PCA(BaseTransformation):
    def __init__(self, n_components=None):
        """
        Creates a PCA in a pipeline with methods to plot elbow & interactive graph.
        :param n_components: The number of components to include in the PCA.
        """
        super().__init__(n_components)
        self.model = SklearnPCA(n_components=self.n_components)
        self.model_name = "PCA"

    def fit(self, X, y=None):
        """
        Overwrites fit method using PCA model.

        :param X: Data to fit.
        :param y: Target variable.
        :return: self for the pipeline.
        """
        self.model.fit(X)
        return self

    def transform(self, X):
        """
        Overwrites transform method using PCA model.

        :param X: Data to fit.
        :param y: Target variable.
        :return: Transform for the pipeline.
        """
        return self.model.transform(X)

    def elbow_plot(self):
        """
        Plots an elbow graph to visualize the optimal number of components to capture important features.
        """
        explained_variances = self.model.explained_variance_ratio_
        cumulative_variances = np.cumsum(explained_variances)

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Different PCA Components')
        plt.grid(True)
        plt.show()