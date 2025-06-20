# src/config.py - Project configuration
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" 
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_DIR = DATA_DIR / "training"
MODELS_DIR = PROJECT_ROOT / "models"

# GCP Configuration - using your existing project
GCP_PROJECT_ID = "guess-the-elo"  # Replace with your actual project ID
GCS_BUCKET_NAME = f"{GCP_PROJECT_ID}-chess-ml-data"
GCS_MODEL_PATH = "models/"

# Model configuration
MODEL_VERSION = "v1"
RANDOM_STATE = 42

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8080

print(f"✅ Chess ELO Predictor configured for project: {GCP_PROJECT_ID}")