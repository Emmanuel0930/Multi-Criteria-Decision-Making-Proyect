"""
feature_engineering.py
=======================
Computes spatial features for each H3 hexagon.

Features are loaded directly from a SQLite database.
"""

from __future__ import annotations

import pandas as pd
import sqlite3


# -------------------------------------------------------------------
# DATABASE LOADER
# -------------------------------------------------------------------

def load_features_from_db(db_path: str) -> pd.DataFrame:
    """
    Load hexagon features from SQLite database.

    Expected table:
        hexagon_features
    """

    conn = sqlite3.connect(db_path)

    df = pd.read_sql(
        "SELECT * FROM hexagon_features",
        conn
    )

    conn.close()

    return df


# -------------------------------------------------------------------
# MAIN FEATURE FUNCTION
# -------------------------------------------------------------------

def engineer_features(
    hex_grid: pd.DataFrame,
    db_path: str
) -> pd.DataFrame:
    """
    Load spatial features from database and attach them to the hex grid.
    """

    print("[Features] Loading features from database...")

    db_df = load_features_from_db(db_path)

    # merge features with hex grid
    df = hex_grid.merge(
        db_df,
        on="hex_id",
        how="left"
    )

    print(f"[Features] Feature matrix shape: {df.shape}")

    return df