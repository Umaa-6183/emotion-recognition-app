# merge_text_datasets.py

import pandas as pd
import os

df = pd.read_csv("data/text_emotion.csv")
print(df.head(10))
print("\n[DEBUG] Empty texts:", df["text"].isnull().sum())
print("[DEBUG] Empty labels:", df["label"].isnull().sum())
print("[DEBUG] Sample labels:", df["label"].unique()[:5])

# Paths
data_dir = "C:/Users/umaam/Downloads/archive"
train_path = os.path.join(data_dir, "train.txt")
test_path = os.path.join(data_dir, "test.txt")
val_path = os.path.join(data_dir, "val.txt")


def load_dataset(path):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ";" in line:
                parts = line.split(";")
                if len(parts) == 2:
                    text, label = parts
                    texts.append(text)
                    labels.append(label)
    return pd.DataFrame({"text": texts, "label": labels})


print("[INFO] Loading datasets...")
train_df = load_dataset(train_path)
test_df = load_dataset(test_path)
val_df = load_dataset(val_path)

print(f"[INFO] Train: {len(train_df)} samples")
print(f"[INFO] Test: {len(test_df)} samples")
print(f"[INFO] Validation: {len(val_df)} samples")

# Combine all
combined_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

# Drop any blank rows
combined_df.dropna(subset=["text", "label"], inplace=True)
combined_df = combined_df[combined_df["text"].str.strip() != ""]

# Save merged CSV
os.makedirs("data", exist_ok=True)
output_path = "data/text_emotion.csv"
combined_df.to_csv(output_path, index=False)

print(f"[INFO] Cleaned dataset saved to: {output_path}")
