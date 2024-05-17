from .base_transformer import BaseTransformation
from sklearn.manifold import TSNE as SklearnTSNE

class TSNE(BaseTransformation):
    def __init__(self, n_components=None, perplexity=30, n_iter=1000, early_exaggeration=12.0):
        """
        Creates a t-SNE in a pipeline with methods to plot graphs.

        :param n_components: The number of components to include in the t-SNE.
        :param perplexity: The perplexity of the t-SNE on the input data.
        :param n_iter: The number of iterations of the t-SNE on the input data.
        """
        super().__init__(n_components)
        self.model_name = "t-SNE"

        self.perplexity = perplexity
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.model = SklearnTSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, early_exaggeration=early_exaggeration)

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
