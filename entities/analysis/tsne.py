import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.cluster import KMeans

from entities.preprocessing.scaling import stats_scaling

class Tsne:
    def __init__(self, df, n_components=2, perplexity=30, n_iter=3000, **kwargs):
        """
        Initializes the Tsne class.

        :param df: The pandas DataFrame.
        :param n_components: The number of components to reduce the data to, typically 2 for visualization.
        :param perplexity: The perplexity parameter for t-SNE.
        :param n_iter: The number of iterations for optimization.
        :param **kwargs: Arbitrary keyword arguments representing groups of features to be used in t-SNE.
        """
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

    def plot(self):
        """
        Plots the t-SNE results.
        """
        plt.figure(figsize=(10, 8))
        colors = cm.rainbow(np.linspace(0, 1, len(np.unique(self.y))))
        for color, label in zip(colors, np.unique(self.y)):
            plt.scatter(self.X_r[self.y == label, 0], self.X_r[self.y == label, 1], color=color, label=label, alpha=0.5)

        plt.legend()
        plt.title('t-SNE Scatter Plot')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

    def plot_interactive(self):
        """
        Plots an interactive scatter plot of the t-SNE results.
        """
        tsne_df = pd.DataFrame(data=self.X_r, columns=['t-SNE Component 1', 't-SNE Component 2'])
        tsne_df['Banned'] = self.y
        tsne_df['pid'] = self.df['pid']

        fig = px.scatter(tsne_df, x='t-SNE Component 1', y='t-SNE Component 2', color='Banned',
                         title='Interactive t-SNE Plot', hover_data=[self.df.columns[0]])
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(template="plotly_dark")
        fig.show()

