import os
import pandas as pd
import logging
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def aggregate_predictions(output_dir, output_file="final_aggregated_predictions.csv"):
    """
    Aggregates all model prediction outputs into a single CSV file.
    Assumes each model writes a CSV with a unique prediction column (e.g., prediction_random_forest).
    Output is written to config['output_dir'].
    """
    logging.basicConfig(level=logging.INFO)
    try:
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(output_dir) if f.startswith("predictions_") and f.endswith(".csv")]
        if not files:
            logging.error("No prediction files found for aggregation.")
            return
        base = pd.read_csv(os.path.join(output_dir, files[0]))
        for f in files[1:]:
            df = pd.read_csv(os.path.join(output_dir, f))
            for col in df.columns:
                if col.startswith("prediction_") and col not in base.columns:
                    base[col] = df[col]
        out_path = os.path.join(output_dir, output_file)
        base.to_csv(out_path, index=False)
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
