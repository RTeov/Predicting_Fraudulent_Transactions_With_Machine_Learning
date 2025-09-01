

import yaml
import os

def load_config(config_path="../config.yaml"):
    with open(os.path.join(os.path.dirname(__file__), config_path), "r") as f:
        return yaml.safe_load(f)

# Placeholder for batch or script entrypoint
if __name__ == "__main__":
    config = load_config()
    print("Loaded config:")
    print(config)
    # Example: access config values
    # input_data = config["input_data"]
    # model_path = config["model_path"]
    # Implement your batch logic here using config values
