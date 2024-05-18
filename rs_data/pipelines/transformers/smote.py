from imblearn.over_sampling import SMOTE as ImblearnSMOTE
from .base_transformer import BaseTransformation
import numpy as np


class SMOTE(BaseTransformation):
    def __init__(self, n_components=None, random_state=42, k_neighbors=5):
        """
        Creates a SMOTE instance to handle imbalanced data, automatically correcting negative values.
        :param n_components: Not used here but kept for compatibility with BaseTransformation.
        :param random_state: The seed used by the random number generator.
        :param k_neighbors: The number of nearest neighbors to used to construct synthetic samples.
        """
        super().__init__(n_components)
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.model = ImblearnSMOTE(random_state=random_state, k_neighbors=k_neighbors)
        self.model_name = "SMOTE"

    def fit(self, X, y=None):
        """
        Fits the SMOTE model on the data, adjusting negative values to zero.

        :param X: Data to fit.
        :param y: Target variable, which dictates the sampling strategy.
        :return: self for the pipeline.
        """
        # Adjust negative values to zero
        X[X < 0] = 0
        self.model.fit_resample(X, y)
        return self

    def transform(self, X):
        """
        Since SMOTE does not transform data but resamples it, this method should not be used.
        Override to prevent misuse.
        """
        return X
    def fit_transform(self, X, y=None):
        """
        Applies the SMOTE resampling to the dataset after adjusting negative values to zero.

        :param X: Data to fit.
        :param y: Target variable.
        :return: The resampled dataset.
        """
        # Adjust negative values to zero
        X[X < 0] = 0

        X_res, y_res = self.model.fit_resample(X, y)
        # Store the components and labels for plotting or other purposes
        self.components_ = X_res
        self.labels_ = y_res

        return X_res, y_res
