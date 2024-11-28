import json
import cv2
import os
import numpy as np
from augmint import Augments  # Assuming you have the Augments class for transformations
from tqdm import tqdm  # For progress bar

# Path to COCO dataset annotations and images
ANNOTATION_PATH = "F:/Kuliah/Semester 5/Kecebut/dataset/Annotations.json"
IMAGE_DIR = "F:/Kuliah/Semester 5/Kecebut/dataset/images"
OUTPUT_DIR = "F:/Kuliah/Semester 5/Kecebut/dataset/dataset_tes/images"
OUTPUT_ANNOTATION_PATH = "F:/Kuliah/Semester 5/Kecebut/dataset/dataset_tes/augmented_val.json"

# Parameters
num_augmented_copies = 1  # Number of augmented copies per image

# Load COCO annotations
with open(ANNOTATION_PATH, "r") as f:
    coco_data = json.load(f)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if COCO data is loaded properly
if 'images' not in coco_data or 'annotations' not in coco_data:
    print("Error: Invalid COCO data format. Missing 'images' or 'annotations'.")
    exit()

# Initialize lists to store augmented images and annotations
augmented_images = []
augmented_annotations = []
annotation_id_counter = max(anno['id'] for anno in coco_data['annotations']) + 1

# Initialize augmentation pipeline
augmenter = Augments()
augmenter.add("rotate", p=0.5, limit=30)
augmenter.add("brightness_contrast", p=0.5, brightness_limit=0.2, contrast_limit=0.2)
augmenter.add("blur", p=0.3, blur_limit=5)
augmenter.add("rgb_shift", p=0.4, r_shift_limit=40, g_shift_limit=30, b_shift_limit=50)

# Augmentation process
for image_info in tqdm(coco_data['images'], desc="Processing Images"):
    img_path = os.path.join(IMAGE_DIR, image_info['file_name'])
    img = cv2.imread(img_path)

    # Check if image was loaded
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue

    # Filter annotations for the current image
    annos = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_info['id']]
    if not annos:
        print(f"Warning: No annotations found for image {image_info['file_name']}. Skipping.")
        continue

    bboxes = [list(map(float, anno['bbox'])) for anno in annos]  # Ensure bboxes are in float format
    category_ids = [anno['category_id'] for anno in annos]

    # Generate multiple augmented copies
    for i in range(num_augmented_copies):
        # Apply augmentations
        aug_img, aug_bboxes = augmenter.apply_augmentations(img, bboxes=bboxes)

        # Generate unique image ID and filename
        new_image_id = image_info['id'] + 10000 + i  # Ensure unique ID
        output_filename = f"augmented_{image_info['id']}_{i}.jpg"
        aug_img_path = os.path.join(OUTPUT_DIR, output_filename)

        # Save augmented image
        if aug_img is not None and len(aug_img.shape) == 3:  # Check for valid image data
            saved = cv2.imwrite(aug_img_path, aug_img)
            if not saved:
                print(f"Error: Failed to save image at {aug_img_path}")
        else:
            print("Error: Augmented image data is invalid.")
            continue

        # Add new image info to augmented images list
        new_image_info = {
            'id': new_image_id,
            'file_name': output_filename,
            'width': aug_img.shape[1],
            'height': aug_img.shape[0]
        }
        augmented_images.append(new_image_info)

        # Add augmented annotations
        for bbox, category_id in zip(aug_bboxes, category_ids):
            new_annotation = {
                'id': annotation_id_counter,
                'image_id': new_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],  # area = width * height
                'iscrowd': 0
            }
            augmented_annotations.append(new_annotation)
            annotation_id_counter += 1

# Update the COCO data with new images and annotations
coco_data['images'].extend(augmented_images)
coco_data['annotations'].extend(augmented_annotations)

# Save the updated annotations to a new JSON file
with open(OUTPUT_ANNOTATION_PATH, "w") as f:
    json.dump(coco_data, f, indent=4)

print("Augmentation complete. Updated annotations saved.")
