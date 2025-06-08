# Team Name: spaceflights

## Project Overview

This project aims to develop a machine learning pipeline that predicts whether a customer is a repeat buyer based on their past transactions, order details, and behavior. The dataset is derived from the Brazilian Olist e-commerce platform.

**Key objectives:**
1. Identify potential repeat buyers to support targeted marketing or retention strategies.
2. Leverage automated pipelines using Kedro for modularity and reproducibility.
3. Use XGBoost as the primary model due to its superior performance and flexibility.
4. Containerize the solution using Docker, ensuring environment consistency across different systems.

The entire process — from data ingestion, preprocessing, model training, to evaluation — is fully automated and can be reproduced in a single command.

### Business & Project Context

Olist is a Brazilian e-commerce platform, similar to Singapore's Lazada/Shopee, that connects small retailers to customers.

**Buying Process:**
1. Customer orders multiple items from various sellers.
2. Logistics partner ships order; customer gets delivery estimated time arrival (ETA) via order receipt.
3. Order receipts gets stored in a database
4. Upon order delivery, customer completes feedback survey.

Primary Objective: Identify potential repeat buyers to build their customer base and increase future sales revenue

## Folder Structure
<pre> <code> 
team_spaceflights/
├── README.md                     # Project documentation (this file)
└── spaceflights/                # Main Kedro ML project
    ├── .dockerignore
    ├── .gitattributes
    ├── .gitignore
    ├── Dockerfile               # Docker image definition
    ├── pyproject.toml           # Python project configuration
    ├── requirements.txt         # Python dependencies
    ├── run.sh                   # Shell script to execute pipeline
    │
    ├── conf/                    # Kedro configuration files
    │   ├── base/
    │   │   ├── catalog.yml
    │   │   ├── parameters.yml
    │   │   └── model_params/
    │   │       └── best_xgb_params.json
    │   └── local/
    │       └── .gitkeep
    │
    ├── data/                    # Project datasets
    │   ├── raw/
    │   │   ├── olist_customers_dataset.csv
    │   │   ├── olist_geolocation_dataset.csv
    │   │   ├── olist_orders_dataset.csv
    │   │   ├── olist_order_items_dataset.csv
    │   │   ├── olist_order_payments_dataset.csv
    │   │   ├── olist_order_reviews_dataset.csv
    │   │   ├── olist_products_dataset.csv
    │   │   ├── olist_sellers_dataset.csv
    │   │   └── extra_information/
    │   │       ├── olist_datadict.xlsx
    │   │       └── product_category_name_translation.csv
    │   └── cleaned/
    │       └── final_merged_olist_with_geolocationV4.csv
    │
    ├── notebooks/
    │   ├── data_prep/
    │   │   ├── all_dataset_cleaned_final.ipynb   # Overall cleaning and insights done by Zhi Yueh
    │   │   ├── geolocation_and_customers_cleaning.ipynb  #cleaning done by Hao Wen
    │   │   ├── items_and_payments_cleaning.ipynb  # cleaning done by Zhi Yueh
    │   │   ├── product_and_seller_cleaning.ipynb  # cleaning done by Hui Ya
    │   │   └── reviews_and_orders_cleaning.ipynb # cleaning done by Claris
    │   └── model_training/
    │       └── XGBoost_Model.ipynb
    │
    ├── saved_model/
    │   └── saved_model_XGBoost_final.pkl
    │
    └── src/
        └── spaceflights/
            ├── __init__.py
            ├── __main__.py
            ├── settings.py
            ├── pipeline_registry.py
            └── pipelines/
                ├── __init__.py
                ├── nodes.py
                └── pipeline.py
</code> </pre>

## Programming Environment
- Language: Python 3.10.12
- Operating System: Windows 11 (WSL2 compatible)
- Framework: Kedro 0.19.12

## EDA Summary & Key Findings

1. Key Findings:
- No correlation between product details (volume or weight) and number of sales 
- No correlation between between distance from customer to seller and number of days the order took to arrive
- No significant variation in customer satisfaction across payment methods
  - Since payment_type does not show meaningful differentiation in review scores, it is unlikely to improve model performance.
  - Keeping it may introduce noise or redundancy rather than useful predictive power.
  
2. Feature Engineering:
- Aggregated customer features by customer_unique_id 
- summed up payment_value, freight_value
- averaged review_score, delivery_time_days
- counted order_item_id
- Label encoding used for categorical customer_state.

## Pipeline Execution Instructions

Execution starts by running:
```bash
bash run.sh
```

**Note: change CRLF to LF in run.sh before execution**

## Overview flow of the pipeline

This project uses Kedro and Docker to manage a complete full end-to-end machine learning pipeline where the entire pipeline is automated and containerized for reproducibility and ease of use.

This script (run.sh) does the following:

- Builds a Docker image containing Python, dependencies, Kedro, and the code (docker build -t spaceflights-image .)
- Runs the image in a container (docker run --rm spaceflights-image)
- Inside the container, the default command kedro run is executed.

Kedro will then executes the pipeline, which runs all registered nodes in the defined order where each node represents a modular step (e.g., preprocessing, training, evaluation).

## Description of logical steps

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

*Raw Data → Preprocessing → Feature Encoding → Split Data → Model Train → Saving Best Parameters → Evaluation → Save*

## Choice of Model

Model Chosen: XGBoost Classifier

Reasoning:
1. State-of-the-art accuracy & generalisation
2. Flexible tuning & regularisation
3. Better Feature Importance Accuracy
4. Used by many award winning competitions

## Model Evaluation
| Metric    | Test Score |
| --------- | ---------- |
| Accuracy  | **0.9872** |
| Precision | **0.8810** |
| Recall    | **0.6610** |
| F1-score  | **0.7553** |

Explanaations:
1. Accuracy shows that the model correctly predicted nearly 99% of all cases.
2. Precision (88.1%) indicates that the model is good at reducing false positives (i.e., when it predicts a repeat buyer, it’s likely to be correct).
3. Recall (66.1%) indicates the model successfully identifies around two-thirds of all actual repeat buyers.
   - Since the goal is to identify potential repeat buyers, recall is a key metric — we prefer to catch more of them, even at the cost of some false positives.
4. F1-score (75.5%) balances both precision and recall, providing a single measure of performance.

*Conclusion: While the model could improve in recall, it already achieves strong overall performance and is well-suited for marketing or retention strategies where capturing more potential repeat buyers is prioritized.*

## Deployment Considerations

1. Model Storage: The trained XGBoost model is saved as a .pkl file (saved_model_XGBoost_final.pkl) and can be loaded directly for inference using joblib or pickle.
2. Environment Reproducibility: A Dockerfile is provided to ensure consistent environment setup. It contains all dependencies and automates execution via run.sh.
3. Input Format Requirement: The model expects input data to match the format and schema used during training. Any additional or missing columns can result in errors unless handled appropriately.
4. Threshold Configuration: The model uses a custom probability threshold of 0.45 for classification. Any deployment should retain or explicitly redefine this threshold to maintain consistent behavior.
5. Scalability: This setup supports batch inference. For real-time deployment (e.g., via REST API), additional wrapping (like using FastAPI or Flask) would be needed.
6. Model Updates: If model retraining is required, simply update parameters in conf/base/model_params/best_xgb_params.json and rerun the pipeline using bash run.sh.
7. Security and Privacy: Ensure that any customer-related data is anonymized or encrypted before deployment, especially if integrating with customer-facing platforms or cloud services.
8. Performance Monitoring: To ensure sustained effectiveness over time, consider adding logging or monitoring of prediction results to detect data drift or reduced accuracy.

## Contributions:

Chua Hui Ya (230571H) - XGBoost Model (used model), initialisation of GitHub repository, assisted in kedro pipeline connection, README.md

Tan Hao Wen (234733T) - RandomForest Model (comparison model), assisted in EDA

Lim Xin Jie Claris (231531C) - nodes.py, pipelines.py, dockerfile, kedro pipeline connection, ensuring smooth running process

Chin Zhi Yueh (234961G) - EDA, analysis and insights
