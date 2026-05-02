import pandas as pd


def convert_dataset(input_path, output_path):
    df = pd.read_csv(input_path)

    records = []

    for _, row in df.iterrows():
        disease = row.iloc[0]

        for col in df.columns[1:]:
            if row[col] == 1:
                records.append({
                    "symptom": col,
                    "disease": disease
                })

    # Debug check
    print("Total records:", len(records))

    new_df = pd.DataFrame(records)
    new_df.to_csv(output_path, index=False)

    print(f"✅ Converted dataset saved to {output_path}")


if __name__ == "__main__":
    convert_dataset(
        "data/raw/Final_Augmented_dataset_Diseases_and_Symptoms.csv",
        "data/processed/symptom_disease_graph.csv"
    )