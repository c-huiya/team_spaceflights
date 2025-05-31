# Team Name: spaceflights

This project documents how we build a machine learning pipeline on Visual Studio Code.

We also do Exploratory Data Analysis (EDA) of our dataset so that we can view and analyse relationships between our dataset.

The 'master' branch consists of RandomForest optimised and non-optimised model and its model training notebook. This is solely used for comparison purposes to our XGBoost Model, since XGBoost's model performance was slightly better than RandomForest's model. Taking a look at the Precision-Recall curve, you might notice that RandomForest's performance was better, so why did we choose XGBoost? Although XGBoost looked like it had a slightly worse curve, teh curve only represents how flexible it is to hyperparameter tuning. Since this is final and we are no longer tuning the hyperparameters for both models, we can safely say that XGBoost is better while looking at both  classification reports.


## Contributions:

Hui Ya - XGBoost Model, initialisation of GitHub repository

Hao Wen - RandomForest Model

Claris - nodes.py, pipelines.py, dockerfile

Zhi Yueh - EDA
