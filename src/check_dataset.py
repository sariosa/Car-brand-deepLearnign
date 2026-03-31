from pathlib import Path
from PIL import Image

# Change this path only if your dataset folder has a different name
DATASET_ROOT = Path("data")
IMAGE_DIR = DATASET_ROOT / "image"
SPLIT_DIR = DATASET_ROOT / "train_test_split" / "classification"

def count_jpgs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.jpg"))

def show_sample_images(root: Path, n: int = 5) -> None:
    samples = list(root.rglob("*.jpg"))[:n]

    if not samples:
        print("No JPG images found.")
        return

    print(f"Found at least {len(samples)} sample images:")
    for p in samples:
        print(" -", p)

    first_image = samples[0]
    img = Image.open(first_image)
    print("\nFirst image opened successfully:")
    print("Path:", first_image)
    print("Size:", img.size)
    print("Mode:", img.mode)

def check_split_files(split_dir: Path) -> None:
    train_file = split_dir / "train.txt"
    test_file = split_dir / "test.txt"

    print("\nChecking split files...")
    print("train.txt exists:", train_file.exists())
    print("test.txt exists:", test_file.exists())

    if train_file.exists():
        print("\nFirst 5 lines of train.txt:")
        with open(train_file, "r") as f:
            for i, line in enumerate(f):
                print(line.strip())
                if i >= 4:
                    break

def main() -> None:
    print("Dataset root exists:", DATASET_ROOT.exists())
    print("Image folder exists:", IMAGE_DIR.exists())

    if not IMAGE_DIR.exists():
        print("Image folder not found.")
        print("Make sure your dataset is directly inside the 'data' folder.")
        return

    total_images = count_jpgs(IMAGE_DIR)
    print("Total JPG images found:", total_images)

    show_sample_images(IMAGE_DIR)
    check_split_files(SPLIT_DIR)

if __name__ == "__main__":
    main()