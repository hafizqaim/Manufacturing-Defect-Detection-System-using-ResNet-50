import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuration ---
MVTEC_ROOT_DIR = '/Users/qaim.ali/Downloads/mvtec_anomaly_detection' 

CATEGORIES = [
    'transistor', 'metal_nut', 'zipper', 'hazelnut', 'cable', 'leather',
    'bottle', 'wood', 'capsule', 'pill', 'tile', 'carpet', 'screw',
    'toothbrush', 'grid'
]

OUTPUT_ROOT_DIR = '/Users/qaim.ali/Desktop/Defect Detection/Datasets/my_defect_dataset'

# Split ratios
DEFECTIVE_TRAIN_RATIO = 0.7
DEFECTIVE_VAL_RATIO = 0.15
GOOD_TRAIN_VAL_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# --- Derived Paths ---
output_train_good_dir = os.path.join(OUTPUT_ROOT_DIR, 'train', 'good_product')
output_train_defective_dir = os.path.join(OUTPUT_ROOT_DIR, 'train', 'defective_product')
output_val_good_dir = os.path.join(OUTPUT_ROOT_DIR, 'val', 'good_product')
output_val_defective_dir = os.path.join(OUTPUT_ROOT_DIR, 'val', 'defective_product')
output_test_good_dir = os.path.join(OUTPUT_ROOT_DIR, 'test', 'good_product')
output_test_defective_dir = os.path.join(OUTPUT_ROOT_DIR, 'test', 'defective_product')

# --- Create Output Directories ---
for path in [output_train_good_dir, output_train_defective_dir,
             output_val_good_dir, output_val_defective_dir,
             output_test_good_dir, output_test_defective_dir]:
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")

# --- Collect all 'Good' Images ---
print("\nCollecting all 'Good' images from specified categories...")
all_good_train_images_paths = []
all_good_test_images_paths = []

for category in CATEGORIES:
    mvtec_category_path = os.path.join(MVTEC_ROOT_DIR, category)
    mvtec_train_good_path = os.path.join(mvtec_category_path, 'train', 'good')
    mvtec_test_good_path = os.path.join(mvtec_category_path, 'test', 'good')

    if os.path.exists(mvtec_train_good_path):
        current_good_train = [os.path.join(mvtec_train_good_path, f) for f in os.listdir(mvtec_train_good_path) if f.endswith('.png')]
        all_good_train_images_paths.extend(current_good_train)
    else:
        print(f"Warning: {mvtec_train_good_path} not found. Skipping good train images for {category}.")

    if os.path.exists(mvtec_test_good_path):
        current_good_test = [os.path.join(mvtec_test_good_path, f) for f in os.listdir(mvtec_test_good_path) if f.endswith('.png')]
        all_good_test_images_paths.extend(current_good_test)
    else:
        print(f"Warning: {mvtec_test_good_path} not found. Skipping good test images for {category}.")

print(f"Total good train images collected: {len(all_good_train_images_paths)}")
print(f"Total good test images collected: {len(all_good_test_images_paths)}")

# Split the combined MVTec's training good into our new train and val good
train_good_files, val_good_files = train_test_split(
    all_good_train_images_paths,
    test_size=GOOD_TRAIN_VAL_SPLIT_RATIO,
    random_state=RANDOM_SEED
)

print(f"Moving {len(train_good_files)} good images to {output_train_good_dir}")
for img_path in train_good_files:
    shutil.copy(img_path, os.path.join(output_train_good_dir, os.path.basename(img_path)))

print(f"Moving {len(val_good_files)} good images to {output_val_good_dir}")
for img_path in val_good_files:
    shutil.copy(img_path, os.path.join(output_val_good_dir, os.path.basename(img_path)))

print(f"Moving {len(all_good_test_images_paths)} good images to {output_test_good_dir}")
for img_path in all_good_test_images_paths:
    shutil.copy(img_path, os.path.join(output_test_good_dir, os.path.basename(img_path)))


# --- Collect all 'Defective' Images ---
print("\nCollecting all 'Defective' images from specified categories...")
all_defective_image_paths = []

for category in CATEGORIES:
    mvtec_category_path = os.path.join(MVTEC_ROOT_DIR, category)
    mvtec_test_defective_root = os.path.join(mvtec_category_path, 'test')

    if not os.path.exists(mvtec_test_defective_root):
        print(f"Warning: {mvtec_test_defective_root} not found. Skipping defective images for {category}.")
        continue

    for defect_type_folder in os.listdir(mvtec_test_defective_root):
        if defect_type_folder == 'good': # Skip the 'good' subfolder
            continue
        defect_type_path = os.path.join(mvtec_test_defective_root, defect_type_folder)
        if os.path.isdir(defect_type_path):
            current_defect_images = [os.path.join(defect_type_path, f) for f in os.listdir(defect_type_path) if f.endswith('.png')]
            all_defective_image_paths.extend(current_defect_images)
        else:
            print(f"Warning: {defect_type_path} is not a directory. Skipping.")


print(f"Found {len(all_defective_image_paths)} total defective images across all categories.")

# Split the combined defective images into train, val, and test for the 'defective_product' class
defective_train_val_combined, defective_test_files = train_test_split(
    all_defective_image_paths,
    test_size=(1 - DEFECTIVE_TRAIN_RATIO - DEFECTIVE_VAL_RATIO),
    random_state=RANDOM_SEED
)

defective_train_files, defective_val_files = train_test_split(
    defective_train_val_combined,
    test_size=DEFECTIVE_VAL_RATIO / (DEFECTIVE_TRAIN_RATIO + DEFECTIVE_VAL_RATIO),
    random_state=RANDOM_SEED
)

print(f"Moving {len(defective_train_files)} defective images to {output_train_defective_dir}")
for img_path in defective_train_files:
    shutil.copy(img_path, os.path.join(output_train_defective_dir, os.path.basename(img_path)))

print(f"Moving {len(defective_val_files)} defective images to {output_val_defective_dir}")
for img_path in defective_val_files:
    shutil.copy(img_path, os.path.join(output_val_defective_dir, os.path.basename(img_path)))

print(f"Moving {len(defective_test_files)} defective images to {output_test_defective_dir}")
for img_path in defective_test_files:
    shutil.copy(img_path, os.path.join(output_test_defective_dir, os.path.basename(img_path)))

print("\n--- Multi-category dataset organization complete ---")
print(f"Check the structure in: {OUTPUT_ROOT_DIR}")

# Final check of counts
print("\n--- Final Counts ---")
print(f"Train Good: {len(os.listdir(output_train_good_dir))}")
print(f"Train Defective: {len(os.listdir(output_train_defective_dir))}")
print(f"Validation Good: {len(os.listdir(output_val_good_dir))}")
print(f"Validation Defective: {len(os.listdir(output_val_defective_dir))}")
print(f"Test Good: {len(os.listdir(output_test_good_dir))}")
print(f"Test Defective: {len(os.listdir(output_test_defective_dir))}")