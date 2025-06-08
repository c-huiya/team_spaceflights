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

The 'master' branch consists of RandomForest optimised and non-optimised model and its model training notebook. This is solely used for comparison purposes to our XGBoost Model, since XGBoost's model performance was better than RandomForest's model. 

The 'main-backup' branch allows us to revert back to a cleaner version of this current repository in case of accidental deletion of the branch.

```bash
bash run.sh
```
*this runs the kedro pipeline*

*Note: change CRLF to LF in run.sh before execution*

## Overview flow of the pipeline

This project uses Kedro and Docker to manage a complete full end-to-end machine learning pipeline where the entire pipeline is automated and containerized for reproducibility and ease of use.

Execution starts by running:
```bash
bash run.sh
```
This script (run.sh) does the following:

- Builds a Docker image containing Python, dependencies, Kedro, and the code (docker build -t spaceflights-image .)
- Runs the image in a container (docker run --rm spaceflights-image)
- Inside the container, the default command kedro run is executed.

Kedro will then executes the pipeline, which runs all registered nodes in the defined order where each node represents a modular step (e.g., preprocessing, training, evaluation).

## Description of logical steps:

The kedro pipeline executes the following steps in sequence:

1. 1_preprocessing

- Merges raw CSV datasets (orders, order_items, payments, etc.).
- Cleans timestamps, removes nulls, and computes derived features (e.g., delivery_time_days).
- Labels each customer as a repeat buyer (repeat_buyer = 1 if more than one order).
- Aggregates features per customer (e.g., total spent, avg review score).

2. encode_customer_state

- Converts the customer_state categorical column into numerical labels using LabelEncoder.

3. 2_split_data

- Splits the processed dataset into training and testing sets (70/30) using stratified sampling based on the target label.

4. 3_train_model

- Trains an XGBoost classifier using hyperparameters loaded from a JSON config file.
- Removes object-type features before training.

5. 4_save_best_params

- Saves the loaded hyperparameters used in training into conf\base\model_params\best_xgb_params.json .

6. 5_evaluate_model

- Predicts on the test set using a probability threshold of 0.45.
- Calculates and returns metrics: Accuracy, Precision, Recall, F1 Score, and AUC-ROC.

7. 6_save_model

- Saves the trained XGBoost model into saved_model/saved_model_XGBoost_final.pkl for later use.

By running bash run.sh, the system builds a Docker image, executes kedro run inside the container, and runs the ML pipeline from data ingestion to model evaluation and saving—all in a fully automated, modular structure.

## Contributions:

Hui Ya - XGBoost Model (used model), initialisation of GitHub repository, assisted in kedro pipeline connection

Hao Wen - RandomForest Model (comparison model), assisted in EDA

Claris - nodes.py, pipelines.py, dockerfile, kedro pipeline connection, ensuring smooth running process

Zhi Yueh - EDA, analysis and insights
