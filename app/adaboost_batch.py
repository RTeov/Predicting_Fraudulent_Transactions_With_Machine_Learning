import os
import yaml
import pandas as pd
import joblib
import boto3
from sklearn.ensemble import AdaBoostClassifier

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
        local_path = "/tmp/output.csv"
        df.to_csv(local_path, index=False)
        bucket, key = output_path.replace("s3://", "").split("/", 1)
        upload_to_s3(local_path, bucket, key)
    else:
        df.to_csv(output_path, index=False)

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

if __name__ == "__main__":
    config = load_config("config.yaml")
    input_path = config["input_data"]
    output_path = config["yaml_output"]
    model_path = config["model_path"]

    predictors = [
        'Transaction_Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
        'Transaction_Amount'
    ]

    df = load_data(input_path)
    model = load_model(model_path)
    df["prediction"] = predict(model, df, predictors)
    save_data(df, output_path)
