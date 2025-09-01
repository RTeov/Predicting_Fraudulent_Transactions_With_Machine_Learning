
import os
import pandas as pd
import logging
import yaml
import tempfile
import boto3

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def aggregate_predictions(output_dir, output_file="final_aggregated_predictions.csv"):
    """
    Aggregates all model prediction outputs into a single CSV file.
    Supports both local directories and S3 URIs for input/output.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        is_s3 = output_dir.startswith("s3://")
        s3 = boto3.client("s3") if is_s3 else None
        temp_dir = tempfile.mkdtemp() if is_s3 else output_dir
        if is_s3:
            bucket, prefix = output_dir.replace("s3://", "").split("/", 1)
            # List all prediction files in the S3 prefix
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv') and os.path.basename(obj['Key']).startswith('predictions_')]
            if not files:
                logging.error("No prediction files found for aggregation in S3.")
                return
            # Download all files to temp_dir
            local_files = []
            for key in files:
                local_path = os.path.join(temp_dir, os.path.basename(key))
                s3.download_file(bucket, key, local_path)
                local_files.append(local_path)
        else:
            os.makedirs(temp_dir, exist_ok=True)
            local_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith("predictions_") and f.endswith(".csv")]
            if not local_files:
                logging.error("No prediction files found for aggregation.")
                return
        base = pd.read_csv(local_files[0])
        for f in local_files[1:]:
            df = pd.read_csv(f)
            for col in df.columns:
                if col.startswith("prediction_") and col not in base.columns:
                    base[col] = df[col]
        out_path = os.path.join(temp_dir, output_file)
        base.to_csv(out_path, index=False)
        if is_s3:
            s3.upload_file(out_path, bucket, os.path.join(prefix, output_file))
            logging.info(f"Aggregated predictions saved to s3://{bucket}/{os.path.join(prefix, output_file)}")
        else:
            logging.info(f"Aggregated predictions saved to {out_path}")
    except Exception as e:
        logging.error(f"Error during aggregation: {e}")

if __name__ == "__main__":
    config = load_config("config.yaml")
    # Optionally set AWS credentials from config if present
    aws_creds = config.get("aws_credentials", {})
    if aws_creds:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_creds.get("aws_access_key_id", "")
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_creds.get("aws_secret_access_key", "")
        if aws_creds.get("aws_session_token"):
            os.environ["AWS_SESSION_TOKEN"] = aws_creds["aws_session_token"]
    output_dir = config.get("output_dir", "./output/")
    aggregate_predictions(output_dir=output_dir)
