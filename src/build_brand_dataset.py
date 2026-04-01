from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT_CSV = Path("outputs/brand_dataset.csv")
OUTPUT_DIR = Path("outputs")
MIN_IMAGES_PER_BRAND = 100

def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    print("Original rows:", len(df))
    print("Original brands:", df["make_id"].nunique())

    # Count images per brand
    brand_counts = df["make_id"].value_counts()

    # Keep only brands with enough images
    valid_brands = brand_counts[brand_counts >= MIN_IMAGES_PER_BRAND].index
    df = df[df["make_id"].isin(valid_brands)].copy()

    print("\nAfter filtering small classes:")
    print("Rows:", len(df))
    print("Brands:", df["make_id"].nunique())

    # Encode labels for deep learning
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["make_id"])

    label_map = pd.DataFrame({
        "make_id": encoder.classes_,
        "label": range(len(encoder.classes_))
    })

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    print("\nTrain rows:", len(train_df))
    print("Test rows:", len(test_df))

    print("\nFirst 10 label mappings:")
    print(label_map.head(10))

    OUTPUT_DIR.mkdir(exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train_brands.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_brands.csv", index=False)
    label_map.to_csv(OUTPUT_DIR / "label_map.csv", index=False)

    print("\nSaved files:")
    print("-", OUTPUT_DIR / "train_brands.csv")
    print("-", OUTPUT_DIR / "test_brands.csv")
    print("-", OUTPUT_DIR / "label_map.csv")

if __name__ == "__main__":
    main()