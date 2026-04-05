import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Configuration
IMAGE_SIZE = 224
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 32
RANDOM_SEED = 42

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def collect_classes(dataset_path):
    """
    Walk the nested dataset structure and collect (class_name -> list of image paths).

    Dataset layout:
        healthy-and-defective-fruits/
            apple_images/
                apple_real_images/      <- fruit_type_real
                    fresh/              <- defect class
                    bruise_defect/
                    ...
                apple_synthethic_images/
                    fresh/
                    ...
            mango_images/
                mango_real_images/
                    fresh/
                    ...
                mango_synthethic_images/
                    ...

    Produces class names like: apple_fresh, apple_bruise_defect, mango_fresh, etc.
    Real and synthetic images for the same defect are merged into the same class.
    """
    class_images = {}  # class_name -> [absolute image paths]

    fruit_dirs = [d for d in Path(dataset_path).iterdir() if d.is_dir()]  # apple_images, mango_images

    for fruit_dir in fruit_dirs:
        image_type_dirs = [d for d in fruit_dir.iterdir() if d.is_dir()]  # real / synthetic

        for image_type_dir in image_type_dirs:
            defect_dirs = [d for d in image_type_dir.iterdir() if d.is_dir()]

            for defect_dir in defect_dirs:
                # Normalise typo: "rot_deffect" -> "rot_defect"
                defect_name = defect_dir.name.replace("deffect", "defect")
                class_name = defect_name

                images = [
                    str(f) for f in defect_dir.iterdir()
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
                ]

                if class_name not in class_images:
                    class_images[class_name] = []
                class_images[class_name].extend(images)

    return class_images


def create_split_directories(output_path, class_names):
    """Create train/val/test directory structure."""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for split in ("train", "val", "test"):
        for class_name in class_names:
            os.makedirs(os.path.join(output_path, split, class_name))


def _copy_and_resize(src_path, dst_path):
    """Copy and resize a single image to IMAGE_SIZE x IMAGE_SIZE."""
    try:
        img = Image.open(src_path).convert("RGB")
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        img_resized.save(dst_path)
    except Exception as e:
        print(f"  Warning: could not process {src_path}: {e}")


def split_and_organize_dataset(dataset_path, output_path):
    """
    Collect all images, split 70/15/15, resize to IMAGE_SIZE, and save to output_path.
    """
    print("Scanning dataset structure...")
    class_images = collect_classes(dataset_path)

    print(f"Found {len(class_images)} classes:")
    for cls, imgs in sorted(class_images.items()):
        print(f"  {cls}: {len(imgs)} images")

    print("\nCreating output directory structure...")
    create_split_directories(output_path, class_images.keys())

    for class_name, images in sorted(class_images.items()):
        train_imgs, temp_imgs = train_test_split(
            images,
            test_size=(VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            random_state=RANDOM_SEED,
        )

        print(f"\nProcessing '{class_name}' — train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}")

        for split, split_imgs in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
            for src in split_imgs:
                filename = os.path.basename(src)
                dst = os.path.join(output_path, split, class_name, filename)
                _copy_and_resize(src, dst)


def get_data_transforms():
    """
    Returns transforms for training and validation/test sets.

    Pipeline:
      - ToTensor()              -> converts PIL image to float tensor, scales [0, 255] -> [0.0, 1.0]
      - Normalize(mean, std)    -> applies ImageNet mean/std normalisation per channel

    Training only: random augmentations applied before ToTensor to improve generalisation
    and handle class imbalance.
    """
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),                                        # [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # ImageNet normalisation
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_test_transform


def create_data_loaders(dataset_path):
    """Create PyTorch DataLoaders for train/val/test sets."""
    train_transform, val_test_transform = get_data_transforms()

    train_dataset = ImageFolder(os.path.join(dataset_path, "train"), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, "val"), transform=val_test_transform)
    test_dataset = ImageFolder(os.path.join(dataset_path, "test"), transform=val_test_transform)

    # num_workers=0 avoids multiprocessing issues on Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes


def save_dataset_info(output_path, classes, class_images_map):
    """Save dataset statistics to a text file."""
    info_path = os.path.join(output_path, "dataset_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"Number of Classes: {len(classes)}\n")
        f.write(f"Classes: {', '.join(sorted(classes))}\n\n")
        f.write(f"Train/Val/Test Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}\n\n")
        f.write("Images per class (total before split):\n")
        for cls in sorted(class_images_map):
            f.write(f"  {cls}: {len(class_images_map[cls])}\n")
        f.write(f"\nNormalisation:\n")
        f.write(f"  ToTensor -> [0.0, 1.0]\n")
        f.write(f"  Normalize mean={IMAGENET_MEAN}, std={IMAGENET_STD}\n")
        f.write(f"\nData Augmentation (training only):\n")
        f.write(f"  - Random Rotation (±20°)\n")
        f.write(f"  - Random Horizontal Flip (p=0.5)\n")
        f.write(f"  - Random Vertical Flip (p=0.2)\n")
        f.write(f"  - Color Jitter (brightness, contrast, saturation ±0.2)\n")
        f.write(f"  - Random Affine Translate (±10%)\n")


def main():
    """Prepare the dataset: split, resize, and set up data loaders."""
    dataset_dir = "healthy-and-defective-fruits"
    output_dir = "prepared_dataset"

    if not os.path.exists(dataset_dir):
        print(f"Error: '{dataset_dir}' not found. Run main.py first to download the dataset.")
        return

    print(f"Input:  {dataset_dir}/")
    print(f"Output: {output_dir}/")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}\n")

    # Collect class -> image paths mapping (needed for info file)
    class_images_map = collect_classes(dataset_dir)

    split_and_organize_dataset(dataset_dir, output_dir)

    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, classes = create_data_loaders(output_dir)

    print(f"\nClasses ({len(classes)}): {classes}")
    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples:       {len(test_loader.dataset)}")

    save_dataset_info(output_dir, classes, class_images_map)
    print(f"\nDataset info saved to: {output_dir}/dataset_info.txt")

    # Verify a batch loads and has the expected shape/range
    sample_batch, _ = next(iter(train_loader))
    print(f"\nSample batch shape: {sample_batch.shape}")
    print(f"Pixel value range (after normalisation): [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
