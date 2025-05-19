#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Preparation Script for ESP32-S3 Motherboard Detection

This script:
1. Creates the directory structure
2. Imports motherboard images
3. Generates synthetic background images
4. Preprocesses and augments all images
5. Splits the dataset into train/val/test sets
6. Visualizes the dataset statistics
"""

import os
import argparse
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_directory_structure(base_dir):
    """Create the necessary directory structure for the project."""
    
    # Main directories
    directories = [
        "raw/motherboard",
        "raw/background",
        "processed/motherboard",
        "processed/background",
        "train/motherboard",
        "train/background",
        "val/motherboard",
        "val/background",
        "test/motherboard",
        "test/background"
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")

def import_images(source_dir, target_dir, category, start_index=1):
    """Import images from source directory to the appropriate raw data directory.
    
    Args:
        source_dir: Directory containing source images
        target_dir: Base data directory
        category: 'motherboard' or 'background'
        start_index: Starting index for renaming files
    """
    
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    
    # Target directory for this category
    target_category_dir = os.path.join(target_dir, "raw", category)
    
    # Get existing files to determine next index
    existing_files = os.listdir(target_category_dir)
    if existing_files:
        existing_indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith(f"{category}_")]
        if existing_indices:
            start_index = max(existing_indices) + 1
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(source_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    
    # Copy and rename files
    for i, filename in enumerate(tqdm(image_files, desc=f"Importing {category} images")):
        source_path = os.path.join(source_dir, filename)
        new_filename = f"{category}_{start_index + i:04d}{os.path.splitext(filename)[1]}"
        target_path = os.path.join(target_category_dir, new_filename)
        
        shutil.copy2(source_path, target_path)
    
    print(f"Imported {len(image_files)} {category} images to {target_category_dir}")

def generate_background_images(data_dir, num_backgrounds=100):
    """Generate background images from existing motherboard images or synthetic patterns."""
    
    motherboard_dir = os.path.join(data_dir, "raw", "motherboard")
    background_dir = os.path.join(data_dir, "raw", "background")
    
    # Ensure the background directory exists
    os.makedirs(background_dir, exist_ok=True)
    
    # Get all motherboard images
    image_files = [f for f in os.listdir(motherboard_dir) 
                  if os.path.splitext(f.lower())[1] in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_files:
        print("No motherboard images found to generate backgrounds from!")
        return
    
    print(f"Generating {num_backgrounds} background images...")
    
    # Get next index for background images
    existing_files = os.listdir(background_dir)
    if existing_files:
        existing_indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith("background_")]
        if existing_indices:
            start_index = max(existing_indices) + 1
        else:
            start_index = 1
    else:
        start_index = 1
    
    # Create backgrounds using various techniques
    for i in tqdm(range(num_backgrounds), desc="Generating backgrounds"):
        idx = i % 6  # Six different methods
        
        if idx == 0:
            # Method 1: Random crop from edge regions
            img_idx = random.randint(0, len(image_files)-1)
            img_path = os.path.join(motherboard_dir, image_files[img_idx])
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Get image dimensions
            h, w = img.shape[:2]
            
            # Determine crop region (avoid center where motherboard likely is)
            side = random.randint(0, 3)  # 0: top, 1: right, 2: bottom, 3: left
            if side == 0:  # top
                crop = img[0:h//3, 0:w]
            elif side == 1:  # right
                crop = img[0:h, 2*w//3:w]
            elif side == 2:  # bottom
                crop = img[2*h//3:h, 0:w]
            else:  # left
                crop = img[0:h, 0:w//3]
            
            # Save crop as background
            output_path = os.path.join(background_dir, f"background_{start_index + i:04d}.jpg")
            cv2.imwrite(output_path, crop)
            
        elif idx == 1:
            # Method 2: Heavy blur + random transformations
            img_idx = random.randint(0, len(image_files)-1)
            img_path = os.path.join(motherboard_dir, image_files[img_idx])
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Apply heavy blur
            blurred = cv2.GaussianBlur(img, (51, 51), 0)
            
            # Apply random transformations
            rows, cols = blurred.shape[:2]
            
            # Create random affine transformation
            M = cv2.getRotationMatrix2D((cols/2, rows/2), random.randint(0, 360), 1.5)
            warped = cv2.warpAffine(blurred, M, (cols, rows))
            
            # Save as background
            output_path = os.path.join(background_dir, f"background_{start_index + i:04d}.jpg")
            cv2.imwrite(output_path, warped)
            
        elif idx == 2:
            # Method 3: Generate solid color
            size = (224, 224)
            texture = np.ones((size[0], size[1], 3), dtype=np.uint8)
            
            # Random color
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            texture[:] = color
            
            # Save as background
            output_path = os.path.join(background_dir, f"background_{start_index + i:04d}.jpg")
            cv2.imwrite(output_path, texture)
            
        elif idx == 3:
            # Method 4: Gradient
            size = (224, 224)
            texture = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            
            # Random gradient direction
            direction = random.randint(0, 3)
            
            for y in range(size[0]):
                for x in range(size[1]):
                    if direction == 0:  # Horizontal
                        val = int(255 * x / size[1])
                    elif direction == 1:  # Vertical
                        val = int(255 * y / size[0])
                    elif direction == 2:  # Diagonal
                        val = int(255 * (x + y) / (size[0] + size[1]))
                    else:  # Radial
                        cx, cy = size[1]//2, size[0]//2
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        max_dist = np.sqrt(cx**2 + cy**2)
                        val = int(255 * dist / max_dist)
                    
                    # Add some noise
                    val = min(255, max(0, val + random.randint(-20, 20)))
                    
                    # Assign random color channels
                    if random.random() < 0.7:  # 70% chance for colored gradient
                        texture[y, x] = (
                            val if random.random() < 0.7 else random.randint(0, 255),
                            val if random.random() < 0.7 else random.randint(0, 255),
                            val if random.random() < 0.7 else random.randint(0, 255)
                        )
                    else:  # 30% chance for grayscale gradient
                        texture[y, x] = (val, val, val)
            
            # Save as background
            output_path = os.path.join(background_dir, f"background_{start_index + i:04d}.jpg")
            cv2.imwrite(output_path, texture)
            
        elif idx == 4:
            # Method 5: Random noise
            size = (224, 224)
            texture = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
            
            # Save as background
            output_path = os.path.join(background_dir, f"background_{start_index + i:04d}.jpg")
            cv2.imwrite(output_path, texture)
            
        else:
            # Method 6: Extreme processing of motherboard images
            img_idx = random.randint(0, len(image_files)-1)
            img_path = os.path.join(motherboard_dir, image_files[img_idx])
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Apply extreme processing
            process_type = random.randint(0, 2)
            
            if process_type == 0:
                # Extreme color manipulation
                # Fix: Convert tuple to list so we can shuffle it
                b, g, r = cv2.split(img)
                channels = [b, g, r]  # Now this is a list, not a tuple
                
                # Shuffle the channels and merge
                random.shuffle(channels)
                processed = cv2.merge(channels)
                
                # Invert colors
                if random.random() < 0.5:
                    processed = cv2.bitwise_not(processed)
                
            elif process_type == 1:
                # Extreme geometric transformations
                rows, cols = img.shape[:2]
                
                # Create a complex geometric transformation
                src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
                dst_points = np.float32([
                    [random.randint(0, cols//3), random.randint(0, rows//3)],
                    [random.randint(2*cols//3, cols-1), random.randint(0, rows//3)],
                    [random.randint(0, cols//3), random.randint(2*rows//3, rows-1)],
                    [random.randint(2*cols//3, cols-1), random.randint(2*rows//3, rows-1)]
                ])
                
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                processed = cv2.warpPerspective(img, M, (cols, rows))
                
            else:
                # Edge detection and color overlay
                edges = cv2.Canny(img, 100, 200)
                processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # Create colored edges
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                # Fix: Create a copy with the same shape and color where the mask is true
                color_overlay = np.zeros_like(processed)
                color_overlay[:] = color
                
                # Apply the color only where edges are detected
                mask = edges > 0
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand mask to 3 channels
                processed = np.where(mask, color_overlay, processed)
            
            # Save as background
            output_path = os.path.join(background_dir, f"background_{start_index + i:04d}.jpg")
            cv2.imwrite(output_path, processed)
    
    print(f"Generated {num_backgrounds} background images in {background_dir}")

def preprocess_images(data_dir, target_size=(224, 224), augment=False):
    """Preprocess raw images and save them to the processed directory.
    
    Args:
        data_dir: Base data directory
        target_size: Size to resize images to (width, height)
        augment: Whether to apply data augmentation
    """
    
    categories = ["motherboard", "background"]
    
    for category in categories:
        raw_dir = os.path.join(data_dir, "raw", category)
        processed_dir = os.path.join(data_dir, "processed", category)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(raw_dir) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        for filename in tqdm(image_files, desc=f"Processing {category} images"):
            # Read image
            img_path = os.path.join(raw_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Could not read image {img_path}")
                continue
            
            # Resize to target size
            img_resized = cv2.resize(img, target_size)
            
            # Save processed image
            processed_path = os.path.join(processed_dir, filename)
            cv2.imwrite(processed_path, img_resized)
            
            # Data augmentation (if enabled)
            if augment:
                base_name = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[1]
                
                # Flip horizontally
                img_flipped = cv2.flip(img_resized, 1)
                flipped_path = os.path.join(processed_dir, f"{base_name}_flip{ext}")
                cv2.imwrite(flipped_path, img_flipped)
                
                # Rotate slightly
                for angle in [10, -10]:
                    h, w = img_resized.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img_rotated = cv2.warpAffine(img_resized, M, (w, h))
                    rotated_path = os.path.join(processed_dir, f"{base_name}_rot{angle}{ext}")
                    cv2.imwrite(rotated_path, img_rotated)
                
                # Adjust brightness
                for factor in [0.8, 1.2]:
                    img_brightness = cv2.convertScaleAbs(img_resized, alpha=factor, beta=0)
                    brightness_path = os.path.join(processed_dir, f"{base_name}_bright{int(factor*100)}{ext}")
                    cv2.imwrite(brightness_path, img_brightness)
    
    print("Preprocessing complete")

def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split processed images into train, validation, and test sets.
    
    Args:
        data_dir: Base data directory
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        print("Error: train_ratio + val_ratio + test_ratio must equal 1")
        return
    
    random.seed(seed)
    categories = ["motherboard", "background"]
    
    for category in categories:
        processed_dir = os.path.join(data_dir, "processed", category)
        train_dir = os.path.join(data_dir, "train", category)
        val_dir = os.path.join(data_dir, "val", category)
        test_dir = os.path.join(data_dir, "test", category)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(processed_dir) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Split indices
        n_files = len(image_files)
        n_train = int(train_ratio * n_files)
        n_val = int(val_ratio * n_files)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]
        
        # Copy files to respective directories
        for filename in train_files:
            shutil.copy2(os.path.join(processed_dir, filename), os.path.join(train_dir, filename))
        
        for filename in val_files:
            shutil.copy2(os.path.join(processed_dir, filename), os.path.join(val_dir, filename))
        
        for filename in test_files:
            shutil.copy2(os.path.join(processed_dir, filename), os.path.join(test_dir, filename))
        
        print(f"Category {category}: {len(train_files)} training, {len(val_files)} validation, {len(test_files)} test images")

def visualize_dataset(data_dir, num_samples=5):
    """Visualize samples from each dataset split and category.
    
    Args:
        data_dir: Base data directory
        num_samples: Number of samples to display per category/split
    """
    
    splits = ["train", "val", "test"]
    categories = ["motherboard", "background"]
    
    plt.figure(figsize=(15, len(splits) * 5))
    
    for i, split in enumerate(splits):
        for j, category in enumerate(categories):
            image_dir = os.path.join(data_dir, split, category)
            image_files = os.listdir(image_dir)
            
            # Randomly select samples
            if len(image_files) > num_samples:
                samples = random.sample(image_files, num_samples)
            else:
                samples = image_files
            
            for k, filename in enumerate(samples):
                if k >= num_samples:
                    break
                    
                img_path = os.path.join(image_dir, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(len(splits), len(categories) * num_samples, 
                           i * (len(categories) * num_samples) + j * num_samples + k + 1)
                plt.imshow(img)
                plt.title(f"{split} - {category}")
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "dataset_samples.png"))
    plt.show()
    print(f"Visualization saved to {os.path.join(data_dir, 'dataset_samples.png')}")

def generate_dataset_stats(data_dir):
    """Generate statistics about the dataset.
    
    Args:
        data_dir: Base data directory
    """
    
    splits = ["train", "val", "test"]
    categories = ["motherboard", "background"]
    
    print("Dataset Statistics:")
    print("-" * 40)
    
    total_images = 0
    
    for split in splits:
        split_total = 0
        for category in categories:
            image_dir = os.path.join(data_dir, split, category)
            if os.path.exists(image_dir):
                num_images = len(os.listdir(image_dir))
                split_total += num_images
                print(f"{split.capitalize()} set - {category}: {num_images} images")
            else:
                print(f"{split.capitalize()} set - {category}: Directory not found")
        
        print(f"{split.capitalize()} set total: {split_total} images")
        print("-" * 40)
        total_images += split_total
    
    print(f"Total dataset size: {total_images} images")

def main():
    """Main function to parse arguments and execute appropriate action."""
    
    parser = argparse.ArgumentParser(description="ESP32-S3 Motherboard Detection Dataset Helper")
    
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Base directory for dataset")
    
    parser.add_argument("--create_dirs", action="store_true",
                       help="Create directory structure")
    
    parser.add_argument("--import_motherboards", type=str, default=None,
                       help="Import motherboard images from specified directory")
    
    parser.add_argument("--generate_backgrounds", type=int, default=None,
                       help="Generate synthetic background images (specify count)")
    
    parser.add_argument("--preprocess", action="store_true",
                       help="Preprocess raw images")
    
    parser.add_argument("--augment", action="store_true",
                       help="Apply data augmentation during preprocessing")
    
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                       help="Target size for preprocessing (width height)")
    
    parser.add_argument("--split", action="store_true",
                       help="Split dataset into train/val/test sets")
    
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize dataset samples")
    
    parser.add_argument("--stats", action="store_true",
                       help="Generate dataset statistics")
    
    parser.add_argument("--all", action="store_true",
                       help="Run all steps in sequence (except importing)")
    
    args = parser.parse_args()
    
    # Execute all steps in sequence if requested
    if args.all:
        create_directory_structure(args.data_dir)
        
        if args.import_motherboards:
            import_images(args.import_motherboards, args.data_dir, "motherboard")
        
        # Generate backgrounds (100 by default if not specified)
        generate_background_images(args.data_dir, 100)
        
        # Preprocess and augment
        preprocess_images(args.data_dir, 
                        target_size=(args.target_size[0], args.target_size[1]),
                        augment=True)
        
        # Split dataset
        split_dataset(args.data_dir)
        
        # Generate statistics
        generate_dataset_stats(args.data_dir)
        
        # Visualize dataset
        visualize_dataset(args.data_dir)
        
        return
    
    # Otherwise execute requested actions individually
    if args.create_dirs:
        create_directory_structure(args.data_dir)
    
    if args.import_motherboards:
        import_images(args.import_motherboards, args.data_dir, "motherboard")
    
    if args.generate_backgrounds is not None:
        generate_background_images(args.data_dir, args.generate_backgrounds)
    
    if args.preprocess:
        preprocess_images(args.data_dir, 
                         target_size=(args.target_size[0], args.target_size[1]),
                         augment=args.augment)
    
    if args.split:
        split_dataset(args.data_dir)
    
    if args.visualize:
        visualize_dataset(args.data_dir)
    
    if args.stats:
        generate_dataset_stats(args.data_dir)
    
    # If no action specified, show help
    if not any([args.create_dirs, args.import_motherboards, args.generate_backgrounds,
               args.preprocess, args.split, args.visualize, args.stats, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()