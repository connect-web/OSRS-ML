from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd

class BaseTransformation(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = None):
        """
        Initialize the transformation with common parameters.

        :param n_components: The number of components to retain.
        """
        self.n_components = n_components
        self.model = None  # This will be set by subclasses
        self.model_name = None

        self.components_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """
        Fit the model to the data. Must be overridden by subclasses.
        This also sets the labels_ variable when parent classes do not have a fit method and this method is invoked.

        :param X: The data to fit.
        :param y The labels to fit.
        """
        if y is not None:
            self.labels_ = y
        return self

    def transform(self, X):
        """
        Transform the data. Must be overridden by subclasses.

        :param X: Data to transform.
        """
        return X

    def fit_transform(self, X, y = None):
        """
        Combines method fit and transform.
        self.labels_ and self.components_ also gets defined here for graphs,
        If method is overwritten then redefine the self.labels_ and self.components_ definitions.
        :param X: Data to transform.
        :param y: The labels to fit.
        :return: Transformed data after fitting model.
        """
        self.fit(X, y)

        if y is not None:
            self.labels_ = y

        self.components_ = self.transform(X)
        return self.components_

    def build_plot_df(self, player_ids: list[int]) -> pd.DataFrame:
        if self.components_ is None or self.labels_ is None:
            raise ValueError("The plot method requires fitted model and labels.")

        plot_df = pd.DataFrame(data=self.components_, columns=[f'{self.model_name} Component {i + 1}' for i in range(self.components_.shape[1])])
        plot_df['Banned'] = self.labels_

        if player_ids is None or len(player_ids) != len(plot_df):
            player_ids = [0] * len(plot_df)

        plot_df['pid'] = player_ids  # Adjust this if 'player_ids' should come from elsewhere
        return plot_df

    def plot(self, player_ids: list[int] = None, x_range: tuple = None, y_range: tuple = None) -> None:
        """
        Plot the results of the transformation using Plotly. This method assumes self.components_
        has been set during transform and that labels and necessary metadata are stored.

        :param player_ids: List of player IDs to include in hover data.
        :param x_range: Range of x-axis.
        :param y_range: Range of y-axis.
        """
        plot_df = self.build_plot_df(player_ids)

        # Using Plotly to create the scatter plot
        fig = px.scatter(plot_df, x=f'{self.model_name} Component 1', y=f'{self.model_name} Component 2', color='Banned',
                         hover_data=['pid'])

        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(template="plotly_dark")

        # Update the x and y range if specified
        if x_range is not None:
            fig.update_xaxes(range=x_range)
        if y_range is not None:
            fig.update_yaxes(range=y_range)

        fig.show()

    def kmeans_cluster(self, n_clusters: int, player_ids: list[int] = None) -> pd.DataFrame:
        """
        Performs K-means clustering and returns a DataFrame with cluster labels.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        clusters = kmeans.fit_predict(self.components_)
        cluster_df = pd.DataFrame(data=self.components_, columns=[f'{self.model_name} Component {i + 1}' for i in range(self.n_components)])
        cluster_df['Banned'] = self.labels_
        cluster_df['Cluster'] = clusters

        if player_ids is None or len(player_ids) != len(cluster_df):
            player_ids = [0] * len(cluster_df)

        cluster_df['pid'] = player_ids

        return cluster_df

    def plot_clusters(self, n_clusters: int, player_ids: list[int] = None):
        """
        Plots results with K-means clusters.
        """
        cluster_df = self.kmeans_cluster(n_clusters, player_ids)
        fig = px.scatter(cluster_df, x=f'{self.model_name} Component 1', y=f'{self.model_name} Component 2', color='Cluster',
                         hover_data=['pid', 'Banned'], title=f'{self.model_name} with {n_clusters} K-means Clusters')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(template="plotly_dark", width=1000, height=800)

        fig.show()
