# Models

Each model is a binary classification model based on specific filters for a SQL Query.
The effect of this is that similar accounts are grouped together therefore the only classification task is to determine is the user should be Banned or not.

I have chosen this specific type of model to focus mainly on the task of determining with the highest possible accuracy whether a user is a bot.

# Model metrics
The model metrics can be seen in the [View](/notebooks/mlflow/experiments/view.ipynb)

- As of right now Mlflow has been run roughly 75 times per Activity across 200+ total activities and will be completing many more runs to grid search and find the best cross validation results across multiple models and hyperparameters.
  - For this reason the model statistics will be stored in the View notebook since they will be changing often.




# Dataset

- The dataset for this project is private and will not be released for the public, however the methods Mlflow has been built on utilizing pandas dataframes therefore allowing classification tasks with different datasets to be utilized from this project.