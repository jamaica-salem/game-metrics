"""
Configuration settings for the gaming analytics project.
Stores database paths, constants, and project-wide settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Database path
DATABASE_PATH = DATA_DIR / "gaming_analytics.db"

# Dataset paths
VGSALES_CSV = RAW_DATA_DIR / "vgsales.csv"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
MODELS_DIR = OUTPUT_DIR / "models"

# Database table names
RAW_TABLE = "raw_games"
CLEANED_TABLE = "cleaned_games"

# Random seed for reproducibility
RANDOM_SEED = 42

# Train-test split ratio
TEST_SIZE = 0.2

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  OUTPUT_DIR, VISUALIZATIONS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)