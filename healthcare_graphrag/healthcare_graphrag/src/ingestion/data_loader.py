"""
data_loader.py
Converts the raw Kaggle binary symptom dataset into a graph-ready
relationship format: symptom → disease (symptom_disease_graph.csv).
"""

import pandas as pd
import os

RAW_PATH = os.path.join("data", "raw", "dataset.csv")
PROCESSED_PATH = os.path.join("data", "processed", "symptom_disease_graph.csv")


def load_raw_dataset(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw Kaggle symptom-disease CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def transform_to_graph_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt the binary symptom matrix into (symptom, disease) edge rows.

    Input columns: Disease, Symptom_1, Symptom_2, ..., Symptom_N  (binary or name)
    Output columns: symptom, disease
    """
    symptom_cols = [c for c in df.columns if c.lower() != "disease"]
    records = []

    for _, row in df.iterrows():
        disease = str(row["Disease"]).strip()
        for col in symptom_cols:
            val = row[col]
            # Support both binary (1/0) and named symptom columns
            if pd.notna(val):
                if isinstance(val, str) and val.strip():
                    records.append({"symptom": val.strip().lower().replace(" ", "_"),
                                    "disease": disease})
                elif isinstance(val, (int, float)) and val == 1:
                    records.append({"symptom": col.strip().lower().replace(" ", "_"),
                                    "disease": disease})

    return pd.DataFrame(records).drop_duplicates()


def save_processed(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data_loader] Saved {len(df)} edges → {path}")


def run():
    print("[data_loader] Loading raw dataset …")
    raw = load_raw_dataset()
    print(f"[data_loader] Raw shape: {raw.shape}")
    graph_df = transform_to_graph_format(raw)
    print(f"[data_loader] Graph edges: {len(graph_df)}")
    save_processed(graph_df)


if __name__ == "__main__":
    run()
