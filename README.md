# About

This project uses machine learning to identify users that are likely to be using macro software on old school runescape.

---

## Contents
- [Production Models - Version 1](#production-models---version-1)
  - [Optimal Model: Performance Metrics](#optimal-model-performance-metrics)
  - [Dataset](#dataset)
  - [Features](#features)
  - [Preprocessor](#preprocessor)
  - [Sampling](#sampling)
  - [Models](#models)
  - [Pipeline](#pipeline)
  

---

### Optimal Model: Performance Metrics

| name                    | version | ROC-AUC  | Mean Accuracy | Recall Class 0 | Recall Class 1 | Accuracy Class 0 | Accuracy Class 1 |
|-------------------------|---------|----------|---------------|----------------|----------------|------------------|------------------|
| prod.Phantom Muspah     | 1       | 0.921394 | 0.929         | 0.947439       | 0.895349       | 0.934            | 0.934            |
| prod.Runecraft          | 1       | 0.919547 | 0.934         | 0.945026       | 0.894068       | 0.933            | 0.933            |
| prod.Artio              | 1       | 0.918723 | 0.928         | 0.933938       | 0.903509       | 0.927            | 0.927            |
| prod.Corporeal Beast    | 1       | 0.915344 | 0.924         | 0.929699       | 0.900990       | 0.921            | 0.921            |
| prod.Zulrah             | 1       | 0.911517 | 0.912         | 0.958781       | 0.864253       | 0.917            | 0.917            |
| prod.Calvarion          | 1       | 0.897222 | 0.935         | 0.944444       | 0.850000       | 0.935            | 0.935            |
| prod.Clue Scrolls (all) | 1       | 0.894737 | 0.906         | 0.909474       | 0.880000       | 0.908            | 0.908            |
| ...                     | ...     | ...      | ...           | ...            | ...            | ...              | ...              |

(Note: Only a subset of the data is shown here for brevity. Please refer to the csv below for complete details.)

[**Metrics CSV - V1**](data/model_metrics/v1/model_metrics.csv)

# Production Models - Version 1



---

### Dataset
The target dataset per activity were grouped by the top 1000 users for the selected activity during the period of November 2023 to January 2024.




### Features
The X features used are as follows:
- ```aggregate_skill_columns```
  - The experience gained for the user across all skills during the aggregated time period.
- ```aggregate_minigame_columns```
  - The minigames and boss scores gained for the user across all minigames / bosses during the aggregated time period.
- ```EXTRA_FEATURES```
  - The extra features within the aggregated period are:
    - Total days checked
    - Total active days
    - Total inactive days
    - Longest active day streak
    - Longest inactive day streak
    - Shortest active day streak
    - Shortest inactive day streak
- ```live_skill_columns```
  - The skill experience that the user ended with at the end of the aggregation period in January 2024.
- ```live_minigame_columns```
  - The minigame scores that the user ended with at the end of the aggregation period in January 2024.

The Y feature is:
- ```Banned```
  - Boolean determining whether a user is banned.

### Preprocessor

```python
preprocessor_default = ColumnTransformer(transformers=[
    ('std', StandardScaler(), aggregate_skill_columns + aggregate_minigame_columns),
    ('robust', RobustScaler(), EXTRA_FEATURES),
    ('minmax', MinMaxScaler(), live_skill_columns),
    ('minmax_2', MinMaxScaler(), live_minigame_columns),
])
```

### Sampling
SMOTE sampling has been used to balance the dataset and improve the metrics for the minority class.

### Models
The models used are as listed below and using the default hyperparameters for version 1.
```python
[
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)),
    ("ExtraTrees", ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42)),  # Does not support n_jobs
    ("SVM", SVC(probability=True, random_state=42)),
    ("LogisticRegression", LogisticRegression(random_state=42, n_jobs=N_JOBS)),
    ("LGBMClassifier", LGBMClassifier(random_state=42, n_jobs=N_JOBS)),
    ("XGBClassifier",XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=N_JOBS))
]
```

### Pipeline

- Roughly 10 runs per Classifier were run to search for the optimal ```pca_n_components``` per activity & per model

Introducing a Bayesian approach for grid-search finding optimal hyper-parameters and ```pca_n_components``` will be used in Version 2 of the models.

```python
pipeline = ImblearnPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('PCA', PCA(n_components=pca_n_components)),
    ('classifier', classifier)
])
```



---
