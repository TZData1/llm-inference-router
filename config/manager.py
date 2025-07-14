# src/config/manager.py
import yaml
import os

def load_config(name):
    """Load config file, return empty dict if fails"""
    path = os.path.join('config', f'{name}.yaml')
    print(f"Loading config from {path}")
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config {path}: {e}")
        return {}
