<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/Pandas-1.3+-yellow?logo=pandas" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-1.21+-orange?logo=numpy" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.0+-blue?logo=scikit-learn" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/LightGBM-3.2+-green?logo=leaflet" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/XGBoost-1.4+-red?logo=xgboost" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/CatBoost-1.0+-yellowgreen?logo=cat" alt="CatBoost"/>
  <img src="https://img.shields.io/badge/Boto3-1.28+-blue?logo=amazon-aws" alt="Boto3"/>
  <img src="https://img.shields.io/badge/AWS%20CLI-required-orange?logo=amazon-aws" alt="AWS CLI"/>
  <img src="https://img.shields.io/badge/Docker-required-blue?logo=docker" alt="Docker"/>
</p>

# Predicting Fraudulent Transactions With Machine Learning

🚀 **A comprehensive machine learning project for real-time fraud detection**

This repository contains a complete end-to-end workflow for detecting fraudulent credit card transactions using advanced machine learning techniques. The project demonstrates industry-standard practices from data preparation through production-ready model deployment, with systematic evaluation of multiple algorithms and robust cross-validation methodology.

## 🏆 Project Highlights

- **🎯 Exceptional Performance**: Achieved high AUC scores with multiple ensemble models
- **📊 Comprehensive Analysis**: 10 detailed Jupyter notebooks covering the complete ML pipeline
- **🔬 Multiple Algorithms**: Systematic evaluation of 5+ machine learning models
- **✅ Production Ready**: Robust cross-validation and optimized hyperparameters
- **📈 Business Impact**: Real-world applicable fraud detection system

---

## 📋 Table of Contents

- [🏆 Project Highlights](#-project-highlights)
- [📖 Project Overview](#-project-overview)
- [📊 Dataset](#-dataset)
- [🔍 Project Structure](#-project-structure)
- [⚙️ Features](#️-features)
- [🔬 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🏅 Results & Performance](#-results--performance)
- [💼 Business Implications](#-business-implications)
- [🚀 How to Run](#-how-to-run)
- [📦 Requirements](#-requirements)
- [🔗 References](#-references)
- [👨‍💻 Author](#-author)

---

## 📖 Project Overview

Credit card fraud detection represents one of the most critical applications of machine learning in the financial industry. This project demonstrates a complete, production-ready approach to building robust fraud detection systems.

### 🎯 Objectives
- **Primary Goal**: Develop a highly accurate fraud detection model with minimal false positives
- **Technical Goal**: Achieve >95% AUC score while maintaining computational efficiency
- **Business Goal**: Create a production-ready system for real-time transaction screening
- **Research Goal**: Compare multiple ML algorithms systematically using cross-validation

### 🔄 Methodology
This project follows industry best practices with a systematic approach:
1. **Comprehensive Data Analysis** - Understanding transaction patterns and fraud characteristics
2. **Feature Engineering** - Correlation analysis and feature selection optimization
3. **Multiple Model Evaluation** - Systematic comparison of 5+ algorithms
4. **Robust Validation** - 5-fold cross-validation for reliable performance estimation
5. **Production Optimization** - Hyperparameter tuning and efficiency optimization

---

## 📊 Dataset

- **Source:** [Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description:** Real anonymized credit card transactions from European cardholders
- **Size:** 284,807 transactions with 31 features
- **Time Period:** September 2013 (2 days of transactions)
- **Class Distribution:** 
  - **Legitimate Transactions:** 99.83% (284,315 transactions)
  - **Fraudulent Transactions:** 0.17% (492 transactions)
- **Challenge:** Highly imbalanced dataset requiring specialized techniques

### 📁 Processed Datasets
- `creditcard.csv` - Original raw dataset
- `creditcard_cleaned.csv` - Post data preparation
- `creditcard_post_exploration.csv` - After exploratory analysis
- `creditcard_post_correlation.csv` - Final optimized feature set

---

## 🔍 Project Structure

```
Credit_Card_Fraud_Detection_Predictive_Model/
│
├── notebooks/                                     # Jupyter notebooks for local development and exploration only
│   ├── 1_Data_Preparation.ipynb                    # Data cleaning & preprocessing
│   ├── 2_Data_Exploration.ipynb                    # EDA & statistical analysis  
│   ├── 3_Features_Correlation.ipynb                # Feature selection & correlation
│   ├── 4_Random Forest Classifier.ipynb            # Baseline ensemble model
│   ├── 5_AdaBoost Classifier.ipynb                 # Adaptive boosting
│   ├── 6_CatBoost Classifier.ipynb                 # Gradient boosting (CatBoost)
│   ├── 7_XGBoost Classifier.ipynb                  # XGBoost implementation
│   ├── 8_LightGBM.ipynb                           # LightGBM single model
│   ├── 9_Training and validation using cross-validation.ipynb  # Cross-validation
│   └── 10_Conclusions and Final Analysis.ipynb     # Results & recommendations
│   └── README.md                                   # Notebooks folder info
│
├── 🗂️ Input_Data/                                  # Dataset storage
│   ├── creditcard.csv                              # Original dataset
│   ├── creditcard_cleaned.csv                      # Processed data
│   ├── creditcard_post_exploration.csv             # Post-EDA data
│   └── creditcard_post_correlation.csv             # Final feature set
│
├── 📊 catboost_info/                               # CatBoost training logs
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   └── time_left.tsv
│
├── 📋 Credit_Card_Fraud_Detection_Predictive_Model.ipynb  # Master notebook
└── 📖 README.md                                    # This file
```

---

## ⚙️ Features

### 📊 Core Transaction Features
- **Transaction_Time:** Seconds elapsed between this transaction and the first transaction in the dataset
- **Transaction_Amount:** Transaction amount in Euros
### 🎯 Feature Engineering Insights
- **Most Important Features:** V17, V12, V14, V10, V11, V16 (identified through feature importance analysis)
- **Correlation Analysis:** Post-correlation feature set optimized for model performance
- **Scaling:** Features properly normalized for optimal model performance

---

## 🔬 Exploratory Data Analysis

The comprehensive EDA includes:

### 📈 Data Quality Assessment
- **Missing Values Analysis:** Complete data integrity verification
- **Duplicate Detection:** Identification and handling of duplicate transactions
- **Outlier Analysis:** Statistical analysis of extreme values in transaction amounts

### 📊 Class Imbalance Analysis
- **Fraud Distribution:** Visualization of class imbalance (0.17% fraud rate)
- **Transaction Patterns:** Time-based analysis of fraud vs. legitimate transactions
- **Amount Analysis:** Statistical comparison of transaction amounts by fraud status

### 🕐 Temporal Analysis
- **Time-based Patterns:** Transaction density analysis by hour
- **Fraud Timing:** Peak fraud detection periods identification
- **Weekly Patterns:** Day-of-week fraud occurrence analysis

### 🔍 Feature Relationships
- **Correlation Heatmaps:** Inter-feature correlation analysis
- **Scatter Plot Analysis:** Fraud pattern visualization in feature space
- **Distribution Analysis:** Feature distribution comparison between fraud/legitimate transactions

---

## 🤖 Machine Learning Models

### 🌳 1. Random Forest Classifier
- **Purpose:** Baseline ensemble model for comparison
- **Strengths:** Feature importance, robust to overfitting
- **Implementation:** Grid search optimization with cross-validation

### ⚡ 2. AdaBoost Classifier  
- **Purpose:** Adaptive boosting for improved performance on imbalanced data
- **Strengths:** Focuses on misclassified examples
- **Optimization:** Learning rate and n_estimators tuning

### 🐱 3. CatBoost Classifier
- **Purpose:** Gradient boosting with efficient categorical feature handling
- **Strengths:** Built-in overfitting protection, minimal hyperparameter tuning
- **Features:** Automatic categorical feature processing

| XGBoost                  | 0.96          | Excellent         | High accuracy, feature importance |

## � How to Run the Project

This section provides a clear, step-by-step guide for running the project both locally and on AWS SageMaker.

---

### 1. Prepare Input Data

- Ensure you have the processed input file: `Input_Data/creditcard_post_correlation.csv`.
- If you do not have this file, generate it by running the following notebooks in order:
  1. `notebooks/1_Data_Preparation.ipynb`
  2. `notebooks/2_Data_Exploration.ipynb`
  3. `notebooks/3_Features_Correlation.ipynb`
- Alternatively, obtain the file from your data pipeline.

---

### 2. Configure the Project

- Edit `config.yaml` to set:
  - `input_data`: Path to your input CSV (local path or S3 URI)
  - `output_dir`: Directory or S3 URI for predictions and results
  - Model paths and AWS credentials as needed
- Make sure your AWS credentials (in `config.yaml` or as environment variables) have permission to access the specified S3 buckets if using cloud features.

---

### 3. Install Dependencies

Install all required Python packages:
```bash
pip install -r requirements.txt
```

---

### 4. Run Batch Inference & Aggregation Locally

Run the batch scripts to generate predictions and aggregate results:
```bash
python app/random_forest_batch.py
python app/adaboost_batch.py
python app/catboost_batch.py
python app/xgboost_batch.py
python app/lightgbm_batch.py
python app/aggregate_results.py
```
All outputs will be saved in the `output_dir` specified in your config.

Or, using Docker:
```bash
docker build -t fraud-batch .
docker run --rm -v %cd%/output:/app/output fraud-batch
```
*(On Windows, use `%cd%` instead of `$(pwd)` for the current directory.)*

---

### 5. Run on AWS SageMaker (Cloud Batch Inference)

1. **Upload input data to S3:**
   - Upload `creditcard_post_correlation.csv` to your S3 bucket.
   - Set `input_data` in `config.yaml` to the S3 URI (e.g., `s3://your-bucket/path/creditcard_post_correlation.csv`).
2. **Build and push Docker image:**
   ```bash
   docker build -t fraud-batch .
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
   docker tag fraud-batch:latest <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest
   ```
3. **Launch a SageMaker Processing or Batch Transform job:**
   - Use the ECR image and specify S3 input/output locations in your SageMaker job definition.
   - The container will run all batch scripts and aggregate results automatically.
   - All outputs will be written to the S3 location mapped to your `output_dir` in `config.yaml`.
   - Pass AWS credentials via `config.yaml` or as environment variables if needed.

---

**Note:**
- All configuration is managed via `config.yaml`.
- For more details on input data format, see the "Example Input Data Format" section below.
- For troubleshooting or advanced usage, refer to the comments in each batch script and the Dockerfile.
   docker run -e AWS_ACCESS_KEY_ID=your-key -e AWS_SECRET_ACCESS_KEY=your-secret -e AWS_SESSION_TOKEN=your-token fraud-batch
   ```

**About `aws_session_token`:**
- If you are using temporary AWS credentials (such as those from an IAM role, federated login, or STS), you must provide `aws_session_token` as well.
- For long-term access keys, leave `aws_session_token` blank or omit it.

**Security Note:**
> Never commit real credentials to version control. Use environment variables or secrets management in production.

---

## 🔧 Batch Inference & Aggregation Workflow

- All batch scripts (Random Forest, AdaBoost, CatBoost, XGBoost, LightGBM) use `config.yaml` for input/output/model paths and AWS credentials.
- Each script writes its predictions to a unique file in the directory specified by `output_dir` in `config.yaml`.
- The aggregation script (`aggregate_results.py`) combines all model predictions into a single CSV in the same `output_dir`.
- You can control all paths and model selection via `config.yaml`.

**Example `config.yaml` structure:**

```yaml
# Optional: Manual AWS credentials for use in Docker/batch scripts
aws_credentials:
   aws_access_key_id: "YOUR_AWS_ACCESS_KEY_ID"
   aws_secret_access_key: "YOUR_AWS_SECRET_ACCESS_KEY"
   aws_session_token: "YOUR_AWS_SESSION_TOKEN"  # Required only for temporary credentials (leave blank for long-term keys)

# Input/output paths for batch script
input_data: "Input_Data/creditcard_post_correlation.csv"  # Local path or S3 URI
output_dir: "output/"    # Directory for all model predictions and aggregation
yaml_output: "output/predictions.csv"    # Default output, but each model will write its own file

# Model settings (set per model if needed)
model_path: "models/random_forest_model.pkl"           # Local path or S3 URI
model_paths:
   random_forest: "models/random_forest_model.pkl"
   adaboost: "models/adaboost_model.pkl"
   catboost: "models/catboost_model.cbm"
   xgboost: "models/xgboost_model.json"
   lightgbm: "models/lightgbm_model.txt"

# AWS settings
aws:
   s3_bucket: "your-s3-bucket-name"

   ---


   **To generate this file:** Run the following notebooks in order:
   1. `1_Data_Preparation.ipynb`
   2. `2_Data_Exploration.ipynb`
   3. `3_Features_Correlation.ipynb`
   These notebooks will process the raw data and produce `creditcard_post_correlation.csv` in the `Input_Data/` folder. Alternatively, obtain the file from your data pipeline if available.

   ---

   ### ▶️ Run Locally (Batch Inference & Aggregation)

   1. **Install dependencies:**
      ```bash
      pip install -r requirements.txt
      ```
   2. **Prepare your input data:**
      - Place your input CSV (with the required columns) in the location specified by `input_data` in `config.yaml`.
   3. **Edit `config.yaml`:**
      - Set all paths, model files, and AWS credentials as needed.
      - Set `output_dir` to the directory where you want all predictions and the aggregated results to be saved.
      - Select which models to run in `models_to_run`.
   4. **Run all batch scripts and aggregate results:**
      ```bash
      python app/random_forest_batch.py
      python app/adaboost_batch.py
      python app/catboost_batch.py
      python app/xgboost_batch.py
      python app/lightgbm_batch.py
      python app/aggregate_results.py
      ```
      Or, if using Docker:
      ```bash
      docker build -t fraud-batch .
      docker run --rm -v $(pwd)/output:/app/output fraud-batch
      ```
      (All outputs will be in the `output_dir` specified in your config.)

   ---

   ### ▶️ Run on AWS SageMaker or Cloud (Batch Inference & Aggregation)

   1. **Upload input data to S3:**
      - Upload `creditcard_post_correlation.csv` to your S3 bucket.
      - In `config.yaml`, set `input_data` to the S3 URI (e.g., `s3://your-bucket/path/creditcard_post_correlation.csv`).
      - Optionally, set `output_dir` to an S3 URI for cloud-based output aggregation.
   2. **Edit `config.yaml`:**
      - Set all model paths, AWS credentials, and other parameters as needed.
      - Ensure your credentials have permission to read/write to the specified S3 locations.
   3. **Build and push Docker image:**
      ```bash
   ## 🚀 How to Run the Project

   This section provides a clear, step-by-step guide for running the project both locally and on AWS SageMaker.

   ---

   ### 1. Prepare Input Data

   - Ensure you have the processed input file: `Input_Data/creditcard_post_correlation.csv`.
   - If you do not have this file, generate it by running the following notebooks in order:
     1. `notebooks/1_Data_Preparation.ipynb`
     2. `notebooks/2_Data_Exploration.ipynb`
     3. `notebooks/3_Features_Correlation.ipynb`
   - Alternatively, obtain the file from your data pipeline.

   ---

   ### 2. Configure the Project

   - Edit `config.yaml` to set:
     - `input_data`: Path to your input CSV (local path or S3 URI)
     - `output_dir`: Directory or S3 URI for predictions and results
     - Model paths and AWS credentials as needed
   - Make sure your AWS credentials (in `config.yaml` or as environment variables) have permission to access the specified S3 buckets if using cloud features.

   ---

   ### 3. Install Dependencies

   Install all required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

   ---

   ### 4. Run Batch Inference & Aggregation Locally

   Run the batch scripts to generate predictions and aggregate results:
   ```bash
   python app/random_forest_batch.py
   python app/adaboost_batch.py
   python app/catboost_batch.py
   python app/xgboost_batch.py
   python app/lightgbm_batch.py
   python app/aggregate_results.py
   ```
   All outputs will be saved in the `output_dir` specified in your config.

   Or, using Docker:
   ```bash
   docker build -t fraud-batch .
   docker run --rm -v %cd%/output:/app/output fraud-batch
   ```
   *(On Windows, use `%cd%` instead of `$(pwd)` for the current directory.)*

   ---

   ### 5. Run on AWS SageMaker (Cloud Batch Inference)

   1. **Upload input data to S3:**
      - Upload `creditcard_post_correlation.csv` to your S3 bucket.
      - Set `input_data` in `config.yaml` to the S3 URI (e.g., `s3://your-bucket/path/creditcard_post_correlation.csv`).
   2. **Build and push Docker image:**
      ```bash
      docker build -t fraud-batch .
      aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
      docker tag fraud-batch:latest <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest
      docker push <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest
      ```
   3. **Launch a SageMaker Processing or Batch Transform job:**
      - Use the ECR image and specify S3 input/output locations in your SageMaker job definition.
      - The container will run all batch scripts and aggregate results automatically.
      - All outputs will be written to the S3 location mapped to your `output_dir` in `config.yaml`.
      - Pass AWS credentials via `config.yaml` or as environment variables if needed.

   ---

   **Note:**
   - All configuration is managed via `config.yaml`.
   - For more details on input data format, see the "Example Input Data Format" section below.
   - For troubleshooting or advanced usage, refer to the comments in each batch script and the Dockerfile.
      # Tag and push
      docker tag fraud-batch:latest <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest
      docker push <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest
      ```
   4. **Launch a SageMaker Processing or Batch Transform job:**
      - Use the ECR image and specify S3 input/output locations in your SageMaker job definition.
      - The container will run all batch scripts and aggregate results automatically (see Dockerfile entrypoint).
      - All outputs will be written to the S3 location mapped to your `output_dir` in `config.yaml`.
      - You can pass AWS credentials via `config.yaml` or as environment variables at job launch if needed.

   ---

---
## 🛠️ Production/Batch Inference & AWS Integration (2025 Update)

## Cloud Deployment

**Note:** The cloud deployment for this project is set up in advance for Amazon SageMaker. All deployment scripts, configurations, and model export formats are designed to be compatible with SageMaker's managed machine learning environment. If you wish to deploy to a different cloud provider, additional modifications may be required.


### 🗂️ Project Structure (Key Folders)
```
app/
   adaboost_batch.py         # AdaBoost batch inference script
   aggregate_results.py      # Aggregates results from all models
   catboost_batch.py         # CatBoost batch inference script
   lightgbm_batch.py         # LightGBM batch inference script
   random_forest_batch.py    # Random Forest batch inference script
   xgboost_batch.py          # XGBoost batch inference script
   __init__.py
   # (main.py has been removed; use the above scripts for batch processing)
   # models/ (if present): model loading utilities

requirements.txt      # Core, ML, and boto3 dependencies
Dockerfile            # Runs batch scripts for inference
.env                  # Environment variables (e.g., model path)
tests/                # Placeholder for batch/boto3-based tests
```

---

## 🚀 Example AWS CLI Commands

**Build, tag, and push your Docker image to ECR:**
```bash
# Build Docker image
docker build -t fraud-batch .

# Authenticate Docker to your ECR registry
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# Tag the image for ECR
docker tag fraud-batch:latest <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest

# Push the image to ECR
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/fraud-batch:latest
```

**Launch a SageMaker Processing or Batch Transform job using your image** (see AWS docs for details).

---

## 📥 Example Input Data Format

Your batch script expects input data as a CSV file with the following columns:

```csv
Transaction_Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Transaction_Amount
12345,0.1,0.2,...,100.0
67890,-0.5,0.3,...,250.0
```
---

## 📦 Requirements

### 🐍 Python Environment
```txt
Python >= 3.7
```

### 📚 Core Libraries
```txt
# Data Processing & Analysis
pandas >= 1.3.0
numpy >= 1.21.0

# Visualization
matplotlib >= 3.4.0
seaborn >= 0.11.0
plotly >= 5.0.0

# Machine Learning
scikit-learn >= 1.0.0

# Gradient Boosting Frameworks
lightgbm >= 3.2.0
xgboost >= 1.4.0
catboost >= 1.0.0

# Jupyter Environment
jupyter >= 1.0.0
ipykernel >= 6.0.0
```

### 💾 System Requirements
- **RAM:** Minimum 8GB (16GB recommended for full dataset)
- **Storage:** ~2GB free space for datasets and model outputs
- **CPU:** Multi-core processor recommended for cross-validation
- **OS:** Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+

### 🔧 Optional Dependencies
```txt
# Enhanced Performance
numba >= 0.53.0      # Accelerated numerical computations
joblib >= 1.0.0      # Parallel processing

# Advanced Visualization  
shap >= 0.39.0       # Model explainability
lime >= 0.2.0        # Local interpretable model explanations

# Model Monitoring
mlflow >= 1.18.0     # Experiment tracking
wandb >= 0.12.0      # Weights & Biases integration
```

---

## 🔗 References

### 📊 Dataset & Research
1. **[Credit Card Fraud Detection Database](https://www.kaggle.com/mlg-ulb/creditcardfraud)** - Anonymized credit card transactions labeled as fraudulent or genuine
2. **[Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)** - Wikipedia comprehensive guide to PCA methodology
3. **[ROC-AUC Characteristics](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)** - Receiver Operating Characteristic and Area Under Curve metrics

### 🤖 Machine Learning Documentation
4. **[Random Forest Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)** - Scikit-learn Random Forest implementation
5. **[AdaBoost Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)** - Scikit-learn AdaBoost documentation
6. **[CatBoost Documentation](https://catboost.ai/)** - Official CatBoost gradient boosting library
7. **[XGBoost Python API](http://xgboost.readthedocs.io/en/latest/python/python_api.html)** - XGBoost Python implementation guide
8. **[LightGBM Python Package](https://github.com/Microsoft/LightGBM/tree/master/python-package)** - Microsoft LightGBM Python implementation
9. **[LightGBM Research Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/lightgbm.pdf)** - Original LightGBM algorithm research paper

### 📚 Additional Resources
10. **[Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)** - Comprehensive machine learning library documentation
11. **[Imbalanced-learn Documentation](https://imbalanced-learn.org/)** - Specialized library for imbalanced dataset handling
12. **[Cross-Validation Techniques](https://scikit-learn.org/stable/modules/cross_validation.html)** - Model validation strategies and best practices

### 🎓 Academic References
- Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. **Calibrating Probability with Undersampling for Unbalanced Classification.** In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
- Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. **Learned lessons in credit card fraud detection from a practitioner perspective**, Expert systems with applications, 41,4, 4915-4928, 2014

---

## 🎯 Project Status

✅ **Complete** - All analysis finished, model optimized, documentation updated

**📊 Final Performance:** 96% AUC Score achieved with XGBoost

**🚀 Production Ready:** Model optimized for real-world deployment

---
