from pathlib import Path
import pandas as pd

DATASET_ROOT = Path("data")
IMAGE_DIR = DATASET_ROOT / "image"
SPLIT_DIR = DATASET_ROOT / "train_test_split" / "classification"
OUTPUT_DIR = Path("outputs")

def read_split_file(file_path: Path, split_name: str) -> list[dict]:
    rows = []

    with open(file_path, "r") as f:
        for line in f:
            relative_path = line.strip()
            if not relative_path:
                continue

            parts = relative_path.split("/")

            if len(parts) != 4:
                continue

            make_id = parts[0]
            model_id = parts[1]
            year = parts[2]
            image_name = parts[3]

            full_image_path = IMAGE_DIR / relative_path

            rows.append({
                "split": split_name,
                "relative_path": relative_path,
                "full_path": str(full_image_path),
                "make_id": int(make_id),
                "model_id": int(model_id),
                "year": year,
                "image_name": image_name,
                "exists": full_image_path.exists()
            })

    return rows

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_file = SPLIT_DIR / "train.txt"
    test_file = SPLIT_DIR / "test.txt"

    train_rows = read_split_file(train_file, "train")
    test_rows = read_split_file(test_file, "test")

    df = pd.DataFrame(train_rows + test_rows)

    print("Total rows:", len(df))
    print("Train rows:", (df["split"] == "train").sum())
    print("Test rows:", (df["split"] == "test").sum())
    print("Unique brands (make_id):", df["make_id"].nunique())
    print("Unique models:", df["model_id"].nunique())
    print("Missing image files:", (~df["exists"]).sum())

    print("\nUnique year examples:")
    print(df["year"].drop_duplicates().sort_values().head(20))

    print("\nFirst 5 rows:")
    print(df.head())

    brand_counts = df["make_id"].value_counts().sort_index()
    print("\nFirst 10 brand counts:")
    print(brand_counts.head(10))

    output_csv = OUTPUT_DIR / "brand_dataset.csv"
    df.to_csv(output_csv, index=False)

    print(f"\nSaved dataset CSV to: {output_csv}")

if __name__ == "__main__":
    main()