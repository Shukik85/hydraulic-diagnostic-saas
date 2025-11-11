"""
Script to check data file structure and identify issues.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_data_file():
    """Check the structure of the data file."""
    data_path = "data/bim_comprehensive.csv"

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    try:
        _extracted_from_check_data_file_11(data_path)
    except Exception as e:
        logger.error(f"Error reading data file: {e}")


# TODO Rename this here and in `check_data_file`
def _extracted_from_check_data_file_11(data_path):
    # Try reading first few rows to understand structure
    df = pd.read_csv(data_path, nrows=5)
    logger.info(f"Data file shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn names:")
    for i, col in enumerate(df.columns):
        print(f"{i:3d}: {col}")

    print("\nData types:")
    print(df.dtypes)

    # Check for expected columns
    expected_components = [
        "pump",
        "cylinder_boom",
        "cylinder_stick",
        "cylinder_bucket",
        "motor_swing",
        "motor_left",
        "motor_right",
    ]

    print("\nMissing expected columns:")
    for component in expected_components:
        expected_cols = [f"{component}_pressure_extend", f"{component}_fault"]
        if missing := [col for col in expected_cols if col not in df.columns]:
            print(f"{component}: {missing}")


if __name__ == "__main__":
    check_data_file()
