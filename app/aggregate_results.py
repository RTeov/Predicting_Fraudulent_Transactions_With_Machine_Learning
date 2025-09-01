import os
import pandas as pd
import logging

def aggregate_predictions(input_path, output_dir, output_file="final_aggregated_predictions.csv"):
    """
    Aggregates all model prediction outputs into a single CSV file.
    Assumes each model writes a CSV with a unique prediction column (e.g., prediction_random_forest).
    """
    logging.basicConfig(level=logging.INFO)
    try:
        # Find all prediction files in output_dir
        files = [f for f in os.listdir(output_dir) if f.startswith("predictions_") and f.endswith(".csv")]
        if not files:
            logging.error("No prediction files found for aggregation.")
            return
        # Load the first file as base
        base = pd.read_csv(os.path.join(output_dir, files[0]))
        # Merge all unique prediction columns
        for f in files[1:]:
            df = pd.read_csv(os.path.join(output_dir, f))
            # Only add new prediction columns
            for col in df.columns:
                if col.startswith("prediction_") and col not in base.columns:
                    base[col] = df[col]
        # Save aggregated file
        out_path = os.path.join(output_dir, output_file)
        base.to_csv(out_path, index=False)
        logging.info(f"Aggregated predictions saved to {out_path}")
    except Exception as e:
        logging.error(f"Error during aggregation: {e}")

if __name__ == "__main__":
    # Example usage: aggregate all predictions in the current directory
    aggregate_predictions(input_path=None, output_dir="./")
