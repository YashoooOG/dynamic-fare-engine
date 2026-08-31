"""
Runner script for Feature Engineering Package.
Executes build_all() to transform interim datasets into ML-ready training partitions.
"""

from src.features.build_features import build_all

if __name__ == "__main__":
    print("Starting feature engineering pipeline...")
    results = build_all()
    print("Feature engineering successfully completed. Processed files written to data/processed/")
