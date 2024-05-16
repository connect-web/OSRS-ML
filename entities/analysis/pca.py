import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
from sklearn.cluster import KMeans
from entities.preprocessing.scaling import stats_scaling

class Pca:
    print_variance = False

    def __init__(self, df, n_components=None, **kwargs):
        """
        Initializes the Pca Class

        :param df: The Pandas Dataframe
        :param n_components: The number of components to create the features with. Default is None for Elbow plot
        :param kwargs: The feature group categories that are columns in the dataframe to be scaled as a group.
        """

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

    def plot(self):
        """
        Plots the first two components of the PCA.
        :return: None
        """
        # Prepare colors - adjust the number of unique labels if needed
        unique_labels = np.unique(self.y)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        # Plotting
        plt.figure(figsize=(10, 8))
        for color, label in zip(colors, unique_labels):
            plt.scatter(self.X_r[self.y == label, 0], self.X_r[self.y == label, 1], c=[color], label=label, alpha=0.4)

        plt.legend()
        plt.title('Scatter plot of PCA components')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def plot_interactive(self, activity=None, x_range = None, y_range = None):
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
        return pca_df

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

    def kmeans_cluster(self, n_clusters):
        """
        Performs K-means clustering on the PCA results and adds the cluster labels to the DataFrame.

        :param n_clusters: The number of clusters to form.
        :return: DataFrame with cluster labels.
        """
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(self.X_r)

        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(data=self.X_r, columns=[f'PC{i + 1}' for i in range(self.X_r.shape[1])])
        pca_df['Banned'] = self.y
        pca_df['Cluster'] = self.df['Cluster']
        pca_df['pid'] = self.df['pid'] if 'pid' in self.df.columns else 'PID not available'

        return pca_df

    def plot_clusters(self, n_clusters):
        """
        Plots the PCA results with K-means clusters.

        :param n_clusters: The number of clusters to form.
        :return: None
        """
        pca_df = self.kmeans_cluster(n_clusters)

        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                         hover_data=['pid', 'Banned'],
                         title=f'PCA with {n_clusters} K-means Clusters')
        fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        fig.update_layout(template="plotly_dark")
        fig.show()
