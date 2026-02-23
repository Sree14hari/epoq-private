"""
Dataset Analyzer - Analyzes image datasets and provides statistics.
"""
import sys
import json
import os
import argparse
from pathlib import Path
from PIL import Image
from collections import Counter


def analyze_dataset(dataset_path):
    """
    Analyzes an image dataset and returns statistics.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary containing dataset statistics
    """
    result = {
        "status": "success",
        "total_images": 0,
        "classes": [],
        "class_counts": {},
        "class_distribution": {},
        "image_sizes": {},
        "avg_image_size": None,
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "splits": {}
    }
    
    if not os.path.exists(dataset_path):
        result["status"] = "error"
        result["message"] = f"Directory not found: {dataset_path}"
        return result
    
    # Check for train/val/test structure
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    test_dir = os.path.join(dataset_path, 'test')
    
    # Also check for 'validation' as alternative
    if not os.path.exists(val_dir):
        val_dir_alt = os.path.join(dataset_path, 'validation')
        if os.path.exists(val_dir_alt):
            val_dir = val_dir_alt
    
    splits = {}
    if os.path.isdir(train_dir):
        splits['train'] = train_dir
    if os.path.isdir(val_dir):
        splits['val'] = val_dir
    if os.path.isdir(test_dir):
        splits['test'] = test_dir
    
    # If no splits, assume flat structure with all images
    if not splits:
        splits = {'': dataset_path}
    
    all_classes = set()
    total_images = 0
    sizes = []
    
    for split_name, split_path in splits.items():
        split_count = 0
        split_sizes = {}
        
        if split_name:
            # This is a split directory (train/val/test)
            if os.path.isdir(split_path):
                for class_name in sorted(os.listdir(split_path)):
                    class_path = os.path.join(split_path, class_name)
                    if os.path.isdir(class_path):
                        all_classes.add(class_name)
                        class_images = [f for f in os.listdir(class_path) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                        class_count = len(class_images)
                        split_count += class_count
                        total_images += class_count
                        
                        # Count sizes for this class
                        for img_file in class_images[:10]:  # Sample first 10 for size info
                            try:
                                img_path = os.path.join(class_path, img_file)
                                with Image.open(img_path) as img:
                                    size = img.size
                                    size_key = f"{size[0]}x{size[1]}"
                                    split_sizes[size_key] = split_sizes.get(size_key, 0) + 1
                                    sizes.append(size)
                            except:
                                pass
        else:
            # No split - check for class folders in root
            if os.path.isdir(split_path):
                for class_name in sorted(os.listdir(split_path)):
                    class_path = os.path.join(split_path, class_name)
                    if os.path.isdir(class_path):
                        all_classes.add(class_name)
                        class_images = [f for f in os.listdir(class_path) 
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                        class_count = len(class_images)
                        split_count += class_count
                        total_images += class_count
                        
                        # Count sizes for this class
                        for img_file in class_images[:10]:
                            try:
                                img_path = os.path.join(class_path, img_file)
                                with Image.open(img_path) as img:
                                    size = img.size
                                    size_key = f"{size[0]}x{size[1]}"
                                    split_sizes[size_key] = split_sizes.get(size_key, 0) + 1
                                    sizes.append(size)
                            except:
                                pass
        
        result["splits"][split_name if split_name else "root"] = split_count
        
        if split_name:
            result[f"{split_name}_count"] = split_count
    
    result["total_images"] = total_images
    result["classes"] = sorted(list(all_classes))
    result["class_count"] = len(all_classes)
    
    # Re-analyze to get class counts per split
    for split_name, split_path in splits.items():
        if os.path.isdir(split_path):
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    class_images = [f for f in os.listdir(class_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
                    key = f"{split_name}/{class_name}" if split_name else class_name
                    result["class_counts"][key] = len(class_images)
    
    # Calculate average image size
    if sizes:
        avg_width = sum(s[0] for s in sizes) / len(sizes)
        avg_height = sum(s[1] for s in sizes) / len(sizes)
        result["avg_image_size"] = f"{int(avg_width)}x{int(avg_height)}"
    
    # Get most common sizes
    if sizes:
        size_counter = Counter(sizes)
        result["common_sizes"] = [
            {"size": f"{s[0]}x{s[1]}", "count": c}
            for s, c in size_counter.most_common(5)
        ]
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Dataset Analyzer')
    parser.add_argument('--path', type=str, required=True, help='Path to dataset')
    args = parser.parse_args()
    
    result = analyze_dataset(args.path)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
