import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
import plotly.express as px

class Pca:
    print_variance = False

    def __init__(self, df, skills, minigames = None, n_components=None):
        """
        Initializes the Pca Class

        :param df: The Pandas Dataframe
        :param skills: Default features
        :param minigames: Optional features
        :param n_components: The number of components to create the features with. Default is None for Elbow plot
        """
        self.df, self.n_components = df, n_components
        self.skills, self.minigames = skills, minigames

        self.y = self.df['Banned']

        self.scaler_skills = StandardScaler()
        self.scaler_minigames = StandardScaler()

    def run(self):
        """
        Scales the data for Skills & Minigames separately
        Runs PCA on the data
        :return:
        """
        scaled_skills = self.scaler_skills.fit_transform(self.df[self.skills])

        # Convert scaled arrays back to DataFrames
        scaled_skills_df = pd.DataFrame(scaled_skills, columns=self.skills)


        # Concatenate scaled dataframes
        if self.minigames is not None:
            scaled_minigames = self.scaler_minigames.fit_transform(self.df[self.minigames])
            scaled_minigames_df = pd.DataFrame(scaled_minigames, columns=self.minigames)
            self.X_scaled = pd.concat([scaled_skills_df, scaled_minigames_df], axis=1)
        else:
            self.X_scaled = scaled_skills_df

        # Setup PCA
        self.pca = PCA(n_components=self.n_components)
        self.X_r = self.pca.fit_transform(self.X_scaled)

        # Explained variance
        if self.print_variance:
            print('explained variance ratio (first two components):', self.pca.explained_variance_ratio_[:2])
            print('sum of explained variance (first two components): ', sum(self.pca.explained_variance_ratio_[:2]))

    def plot(self):
        """
        Plots the PCA
        :return:
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

    def plot_interactive(self, activity=None):
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
        fig.show()
        return pca_df

    def plot_interactive_3d(self):
        pca_df = pd.DataFrame(data=self.X_r, columns=[f'PC{i + 1}' for i in range(self.X_r.shape[1])])
        pca_df['Banned'] = self.y
        pca_df['pid'] = self.df['pid'] if 'pid' in self.df.columns else 'PID not available'

        fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Banned',
                            hover_data=['pid', 'Overall'])
        fig.update_layout(template="plotly_dark")
        fig.show()

    def elbow_plot(self):
        explained_variances = self.pca.explained_variance_ratio_
        cumulative_variances = np.cumsum(explained_variances)

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Different PCA Components')
        plt.grid(True)
        plt.show()