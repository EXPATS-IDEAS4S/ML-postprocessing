import yaml

# === Load YAML Config ===
def load_config(config_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters as a nested dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
