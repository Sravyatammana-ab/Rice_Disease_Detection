"""
Combine RiceDiseaseDataset and LabelledRice datasets into a single unified dataset.
This script merges both datasets, removes duplicates, and creates a stratified train/val split.
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm

# Configuration
DATASET_DIR = Path("dataset")
RICE_DISEASE_DATASET = DATASET_DIR / "RiceDiseaseDataset"
LABELLED_RICE_DATASET = DATASET_DIR / "LabelledRice" / "Labelled"
COMBINED_DATASET = DATASET_DIR / "CombinedDataset"

# Classes
CLASSES = ["BrownSpot", "Healthy", "Hispa", "LeafBlast"]

# Train/Val split ratio
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

def get_file_hash(filepath):
    """Calculate MD5 hash of a file to detect duplicates."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def collect_images_from_dataset(dataset_path, dataset_name):
    """Collect all images from a dataset with their hashes."""
    images = defaultdict(list)  # class_name -> [(path, hash)]
    
    print(f"\n📂 Collecting images from {dataset_name}...")
    
    for class_name in CLASSES:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_dir} does not exist, skipping...")
            continue
        
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG"))
        
        for img_path in tqdm(image_files, desc=f"  {class_name}", leave=False):
            file_hash = get_file_hash(img_path)
            images[class_name].append((img_path, file_hash))
    
    return images

def merge_datasets():
    """Merge both datasets and remove duplicates."""
    print("=" * 70)
    print("🔄 COMBINING RICE DISEASE DATASETS")
    print("=" * 70)
    
    # Collect images from both datasets
    rice_disease_images = {}
    labelled_rice_images = {}
    
    # Collect from RiceDiseaseDataset (train + validation)
    print("\n📊 Dataset 1: RiceDiseaseDataset")
    for split in ["train", "validation"]:
        split_path = RICE_DISEASE_DATASET / split
        if split_path.exists():
            split_images = collect_images_from_dataset(split_path, f"RiceDiseaseDataset/{split}")
            for class_name, images in split_images.items():
                if class_name not in rice_disease_images:
                    rice_disease_images[class_name] = []
                rice_disease_images[class_name].extend(images)
    
    # Collect from LabelledRice
    print("\n📊 Dataset 2: LabelledRice")
    labelled_rice_images = collect_images_from_dataset(LABELLED_RICE_DATASET, "LabelledRice")
    
    # Merge and remove duplicates
    print("\n🔍 Merging datasets and removing duplicates...")
    combined_images = defaultdict(list)
    seen_hashes = defaultdict(set)  # class_name -> set of hashes
    stats = {"total": 0, "duplicates": 0, "unique": 0}
    
    for class_name in CLASSES:
        print(f"\n  Processing {class_name}:")
        
        # Add images from both datasets
        all_images = rice_disease_images.get(class_name, []) + labelled_rice_images.get(class_name, [])
        stats["total"] += len(all_images)
        
        for img_path, file_hash in all_images:
            if file_hash not in seen_hashes[class_name]:
                combined_images[class_name].append(img_path)
                seen_hashes[class_name].add(file_hash)
                stats["unique"] += 1
            else:
                stats["duplicates"] += 1
        
        print(f"    Total: {len(all_images)}, Unique: {len(combined_images[class_name])}, Duplicates: {len(all_images) - len(combined_images[class_name])}")
    
    print(f"\n📈 Overall Statistics:")
    print(f"    Total images processed: {stats['total']}")
    print(f"    Unique images: {stats['unique']}")
    print(f"    Duplicates removed: {stats['duplicates']}")
    
    return combined_images

def create_train_val_split(combined_images):
    """Create stratified train/validation split."""
    print("\n📊 Creating train/validation split (80/20)...")
    
    random.seed(RANDOM_SEED)
    train_images = defaultdict(list)
    val_images = defaultdict(list)
    
    for class_name, images in combined_images.items():
        # Shuffle images
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        # Split
        split_idx = int(len(shuffled) * TRAIN_RATIO)
        train_images[class_name] = shuffled[:split_idx]
        val_images[class_name] = shuffled[split_idx:]
        
        print(f"  {class_name}: {len(train_images[class_name])} train, {len(val_images[class_name])} val")
    
    return train_images, val_images

def copy_images(train_images, val_images):
    """Copy images to the combined dataset directory."""
    print("\n📁 Creating combined dataset structure...")
    
    # Create directory structure
    for split in ["train", "validation"]:
        for class_name in CLASSES:
            split_dir = COMBINED_DATASET / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy training images
    print("\n📥 Copying training images...")
    for class_name, images in train_images.items():
        dest_dir = COMBINED_DATASET / "train" / class_name
        for img_path in tqdm(images, desc=f"  {class_name}", leave=False):
            # Create unique filename to avoid conflicts
            dest_path = dest_dir / img_path.name
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1
            shutil.copy2(img_path, dest_path)
    
    # Copy validation images
    print("\n📥 Copying validation images...")
    for class_name, images in val_images.items():
        dest_dir = COMBINED_DATASET / "validation" / class_name
        for img_path in tqdm(images, desc=f"  {class_name}", leave=False):
            # Create unique filename to avoid conflicts
            dest_path = dest_dir / img_path.name
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1
            shutil.copy2(img_path, dest_path)

def verify_dataset():
    """Verify the combined dataset."""
    print("\n" + "=" * 70)
    print("✅ VERIFICATION")
    print("=" * 70)
    
    total_train = 0
    total_val = 0
    
    print("\n📊 Final Dataset Statistics:")
    print("\n  Training Set:")
    for class_name in CLASSES:
        train_dir = COMBINED_DATASET / "train" / class_name
        count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.JPG")))
        total_train += count
        print(f"    {class_name:12s}: {count:4d} images")
    
    print(f"    {'Total':12s}: {total_train:4d} images")
    
    print("\n  Validation Set:")
    for class_name in CLASSES:
        val_dir = COMBINED_DATASET / "validation" / class_name
        count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.JPG")))
        total_val += count
        print(f"    {class_name:12s}: {count:4d} images")
    
    print(f"    {'Total':12s}: {total_val:4d} images")
    
    print(f"\n  Grand Total: {total_train + total_val} images")
    print(f"  Split Ratio: {total_train/(total_train+total_val)*100:.1f}% train, {total_val/(total_train+total_val)*100:.1f}% val")
    
    # Calculate class weights for binary classification
    print("\n📊 Class Imbalance Analysis for Binary Classification:")
    diseased_train = sum([len(list((COMBINED_DATASET / "train" / c).glob("*.jpg"))) + len(list((COMBINED_DATASET / "train" / c).glob("*.JPG"))) for c in ["BrownSpot", "Hispa", "LeafBlast"]])
    healthy_train = len(list((COMBINED_DATASET / "train" / "Healthy").glob("*.jpg"))) + len(list((COMBINED_DATASET / "train" / "Healthy").glob("*.JPG")))
    
    total_samples = diseased_train + healthy_train
    weight_diseased = total_samples / (2 * diseased_train)
    weight_healthy = total_samples / (2 * healthy_train)
    
    print(f"  Diseased (BrownSpot+Hispa+LeafBlast): {diseased_train} images")
    print(f"  Healthy: {healthy_train} images")
    print(f"\n  Recommended Binary Class Weights:")
    print(f"    Diseased: {weight_diseased:.4f}")
    print(f"    Healthy:  {weight_healthy:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ DATASET COMBINATION COMPLETE!")
    print("=" * 70)

def main():
    """Main function to combine datasets."""
    # Check if combined dataset already exists
    if COMBINED_DATASET.exists():
        response = input(f"\n⚠️  {COMBINED_DATASET} already exists. Overwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Aborted.")
            return
        print("🗑️  Removing existing combined dataset...")
        shutil.rmtree(COMBINED_DATASET)
    
    # Merge datasets
    combined_images = merge_datasets()
    
    # Create train/val split
    train_images, val_images = create_train_val_split(combined_images)
    
    # Copy images
    copy_images(train_images, val_images)
    
    # Verify
    verify_dataset()
    
    print("\n🎉 Next steps:")
    print("  1. Update config.py to use 'CombinedDataset'")
    print("  2. Add class weights to binary_train.py")
    print("  3. Run: python binary_train.py --model ResNet50 --epochs 30")
    print("=" * 70)

if __name__ == "__main__":
    main()
