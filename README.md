# Team Name: spaceflights

## Project Objective

Olist is a Brazilian e-commerce platform, similar to Singapore's Lazada / Shopee application, which connects small retailers to customers.

How it operates:

1. customer chooses what items to order from various sellers
2. customer gets their receipt which is stored in a database
3. customer gets their delivery estimated time arrival (ETA)
4. customer fills up a feedback survey upon order arrival

Project objective: identification of potential repeat buyers

Business objective: improvement of future sales revenue by catering to respective customers 

## What this project contains (what we do in this project)

This project documents how we build a machine learning pipeline on Visual Studio Code.

We also do Exploratory Data Analysis (EDA) of our dataset so that we can view and analyse relationships between our dataset.

The 'master' branch consists of RandomForest optimised and non-optimised model and its model training notebook. This is solely used for comparison purposes to our XGBoost Model, since XGBoost's model performance was slightly better than RandomForest's model. Taking a look at the Precision-Recall curve, you might notice that RandomForest's performance was better, so why did we choose XGBoost? Although XGBoost looked like it had a slightly worse curve, teh curve only represents how flexible it is to hyperparameter tuning. Since this is final and we are no longer tuning the hyperparameters for both models, we can safely say that XGBoost is better while looking at both  classification reports.

```bash
bash run.sh
```
*this runs the kedro pipeline*

*Note: change CRLF to LF in run.sh before execution*


## Contributions:

Hui Ya - XGBoost Model (used model), initialisation of GitHub repository, assisted in kedro pipeline connection

Hao Wen - RandomForest Model (comparison model), assisted in EDA

Claris - nodes.py, pipelines.py, dockerfile, kedro pipeline connection, ensuring smooth running process

Zhi Yueh - EDA, analysis and insights
