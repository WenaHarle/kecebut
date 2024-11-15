import json
import random
import os
import shutil

def split_coco_annotations_fixed(annotations_file, output_dir, train_size=9, val_size=2, test_size=1, seed=42):
    """
    Split COCO dataset into train, validation, and test sets with fixed numbers per class,
    and move images to new directories for each split.
    
    Args:
        annotations_file (str): Path to the COCO annotations JSON file.
        output_dir (str): Path to the directory where split datasets and images will be saved.
        train_size (int): Number of images per class for training.
        val_size (int): Number of images per class for validation.
        test_size (int): Number of images per class for testing.
        seed (int): Random seed for reproducibility.
        
    Returns:
        None
    """
    assert train_size + val_size + test_size == 12, "The split sizes must add up to 12 per class"
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get all images by category
    category_to_images = {}
    for ann in coco_data['annotations']:
        category_id = ann['category_id']
        image_id = ann['image_id']
        if category_id not in category_to_images:
            category_to_images[category_id] = set()
        category_to_images[category_id].add(image_id)
    
    # Split images for each category
    train_images, val_images, test_images = set(), set(), set()
    random.seed(seed)
    for category, images in category_to_images.items():
        images = list(images)
        random.shuffle(images)
        
        train_images.update(images[:train_size])
        val_images.update(images[train_size:train_size + val_size])
        test_images.update(images[train_size + val_size:train_size + val_size + test_size])
    
    # Create split datasets
    def filter_annotations(images_set):
        images = [img for img in coco_data['images'] if img['id'] in images_set]
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in images_set]
        return {
            "images": images,
            "annotations": annotations,
            "categories": coco_data['categories']
        }
    
    train_data = filter_annotations(train_images)
    val_data = filter_annotations(val_images)
    test_data = filter_annotations(test_images)
    
    # Save new annotation files
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train_annotations.json'), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(output_dir, 'val_annotations.json'), 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(os.path.join(output_dir, 'test_annotations.json'), 'w') as f:
        json.dump(test_data, f, indent=4)

    # Create directories for images
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Move images to their respective directories
    def move_images(images_set, target_dir):
        for img in coco_data['images']:
            if img['id'] in images_set:
                # Use the full path from the file_name field
                img_filename = img['file_name']
                src_path = img_filename  # Use the relative path directly from the JSON
                dest_path = os.path.join(target_dir, os.path.basename(img_filename))
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                else:
                    print(f"Warning: Image {src_path} not found.")
    
    move_images(train_images, train_dir)
    move_images(val_images, val_dir)
    move_images(test_images, test_dir)

    print("Splits created and images moved to directories.")

split_coco_annotations_fixed("result.json", "F:/Kuliah/Semester 5/Kecebut/New Dataset", train_size=9, val_size=2, test_size=1)
