from .base_transformer import BaseTransformation
from umap import UMAP as LibraryUMAP  # Make sure UMAP is properly installed and imported


class UMAP(BaseTransformation):
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Creates a UMAP in a pipeline with methods to plot graphs.

        :param n_components: The number of components to include in the UMAP.
        :param n_neighbors: The number of neighbors to include in the UMAP.
        """
        super().__init__(n_components)
        self.model_name = "UMAP"

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

        self.model = LibraryUMAP(n_components=n_components,
                          n_neighbors=n_neighbors,
                          min_dist=min_dist,
                          random_state=random_state)

    def fit_transform(self, X, y=None):
        """
        Fits and transforms for t-SNE
        This is the only method because t-SNE does not separate fit and transform,
        the fit method also sets the labels variable.

        :param X: Data to fit.
        :param y: Target variable.
        :return: Transformed data after fitting model.
        """
        super().fit(X, y)
        self.components_ = self.transform(X)
        return self.components_

    def transform(self, X):
        """
        Overwrites transform method using PCA model.

        :param X: Data to fit.
        :param y: Target variable.
        :return: Transform for the pipeline.
        """
        return self.model.fit_transform(X)