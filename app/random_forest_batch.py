
import os
import yaml
import pandas as pd
import joblib
import boto3
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_from_s3(bucket, key, dest):
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, dest)

def upload_to_s3(src, bucket, key):
    s3 = boto3.client("s3")
    s3.upload_file(src, bucket, key)


def load_data(input_path):
    if input_path.startswith("s3://"):
        bucket, key = input_path.replace("s3://", "").split("/", 1)
        local_path = "/tmp/input.csv"
        download_from_s3(bucket, key, local_path)
        return pd.read_csv(local_path)
    else:
        return pd.read_csv(input_path)


def save_data(df, output_path):
    if output_path.startswith("s3://"):
        local_path = "/tmp/output_random_forest.csv"
        df.to_csv(local_path, index=False)
        bucket, key = output_path.replace("s3://", "").split("/", 1)
        upload_to_s3(local_path, bucket, key)
    else:
        out_path = output_path.replace("predictions", "predictions_random_forest")
        df.to_csv(out_path, index=False)

def load_model(model_path):
    if model_path.startswith("s3://"):
        bucket, key = model_path.replace("s3://", "").split("/", 1)
        local_path = "/tmp/model.pkl"
        download_from_s3(bucket, key, local_path)
        return joblib.load(local_path)
    else:
        return joblib.load(model_path)


def predict(model, df, predictors):
    return model.predict(df[predictors])


def validate_input(df, predictors):
    missing = [col for col in predictors if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

def log_metrics(y_true, y_pred):
    logging.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(y_true, y_pred):.4f}")
    logging.info(f"Recall: {recall_score(y_true, y_pred):.4f}")
    logging.info(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    logging.info(f"ROC-AUC Score: {roc_auc_score(y_true, y_pred):.4f}")
    logging.info("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"]))

def main():
    logging.basicConfig(level=logging.INFO)
    try:
        config = load_config("config.yaml")
        models_to_run = config.get("models_to_run", ["random_forest"])
        if "random_forest" not in models_to_run:
            logging.info("Random Forest skipped by config.")
            return
        input_path = config["input_data"]
        output_path = config["yaml_output"]
        model_path = config["model_path"]

        predictors = [
            'Transaction_Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
            'Transaction_Amount'
        ]

        logging.info("Loading input data...")
        df = load_data(input_path)
        validate_input(df, predictors)
        logging.info("Loading model...")
        model = load_model(model_path)
        logging.info("Running predictions...")
        df["prediction_random_forest"] = predict(model, df, predictors)
        save_data(df, output_path)
        if "Fraud_Flag" in df.columns:
            log_metrics(df["Fraud_Flag"], df["prediction_random_forest"])
        logging.info("Random Forest batch inference complete.")
    except Exception as e:
        logging.error(f"Error in Random Forest batch: {e}")

if __name__ == "__main__":
    main()
