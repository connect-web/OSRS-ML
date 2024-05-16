import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import plotly.express as px

class Plot:
    def __init__(self, df, n_components, **kwargs):
        self.df = df
        self.n_components = n_components
        self.feature_groups = [kwargs[key] for key in kwargs if kwargs[key]]
        self.y = self.df['Banned']  # Assuming 'Banned' is a label column in your DataFrame
        self.model_name = None  # To be defined in subclasses
        self.X_r = None  # Resulting low-dimensional representation

    def run(self):
        raise NotImplementedError("Subclasses should implement this!")

    def plot(self):
        plt.figure(figsize=(10, 8))
        colors = cm.rainbow(np.linspace(0, 1, len(np.unique(self.y))))
        for color, label in zip(colors, np.unique(self.y)):
            plt.scatter(self.X_r[self.y == label, 0], self.X_r[self.y == label, 1], color=color, label=label, alpha=0.5)
        plt.legend()
        plt.title(f'{self.model_name} Scatter Plot')
        plt.xlabel(f'{self.model_name} Component 1')
        plt.ylabel(f'{self.model_name} Component 2')
        plt.show()

    def plot_interactive(self, activity=None, x_range=None, y_range=None, return_df = False):
        """
        Plots an interactive PCA graph with pid's included in points to recognise patterns in the data.

        :param activity: The activity column from the dataframe being plotted.
        :return: None
        """
        # Assuming self.X_r has been computed with n_components at least 2
        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(data=self.X_r, columns=[f'PC{i + 1}' for i in range(self.X_r.shape[1])])
        pca_df['Banned'] = self.y  # Adding the label column for color-coding

        # Optionally add other columns from original df for more detailed tooltips
        # Ensure these columns are in the original dataframe before adding
        pca_df['pid'] = self.df['pid'] if 'pid' in self.df.columns else 'PID not available'
        pca_df['Overall'] = self.df['Overall'] if 'Overall' in self.df.columns else 'Overall not available'

        # If KC is a valid column, include it; otherwise, provide a placeholder

        if activity is None:
            pca_df['activity'] = 'KC not available'
        else:
            pca_df['activity'] = self.df[activity] if activity in self.df.columns else 'KC not available'

        # Plotting using Plotly, focusing on the first two PCs for 2D visualization
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Banned',
                         hover_data=['pid', 'activity', 'Overall'])

        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(template="plotly_dark")

        # Update the x and y range if specified
        if x_range is not None:
            fig.update_xaxes(range=x_range)
        if y_range is not None:
            fig.update_yaxes(range=y_range)

        fig.show()
        if return_df:
            return pca_df

    def kmeans_cluster(self, n_clusters):
        """
        Performs K-means clustering and returns a DataFrame with cluster labels.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.X_r)
        cluster_df = pd.DataFrame(data=self.X_r, columns=[f'{self.model_name} Component {i + 1}' for i in range(self.n_components)])
        cluster_df['Banned'] = self.y
        cluster_df['Cluster'] = clusters
        cluster_df['pid'] = self.df['pid'] if 'pid' in self.df.columns else 'PID not available'
        return cluster_df

    def plot_clusters(self, n_clusters):
        """
        Plots results with K-means clusters.
        """
        cluster_df = self.kmeans_cluster(n_clusters)
        fig = px.scatter(cluster_df, x=f'{self.model_name} Component 1', y=f'{self.model_name} Component 2', color='Cluster',
                         hover_data=['pid', 'Banned'], title=f'{self.model_name} with {n_clusters} K-means Clusters')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(template="plotly_dark", width=1000, height=800)
        fig.show()