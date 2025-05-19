#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Quantization Script for ESP32-S3 Motherboard Detection

This script quantizes an ONNX model using ESP-PPQ and exports it to ESP-DL format
for deployment on ESP32-S3.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from PIL import Image

# Import ESP-PPQ modules
import esp_ppq
from esp_ppq import quantize, write_to_spiffs, evaluate

class CalibrationDataset:
    """Dataset class for collecting calibration data from images."""
    
    def __init__(self, data_dir, img_size=224, max_samples=100):
        self.data_dir = data_dir
        self.img_size = img_size
        self.max_samples = max_samples
        
        # Get image files
        self.image_files = []
        categories = ["motherboard", "background"]
        
        for category in categories:
            category_dir = os.path.join(data_dir, "train", category)
            if os.path.exists(category_dir):
                files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) 
                        if os.path.isfile(os.path.join(category_dir, f)) and 
                        f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                self.image_files.extend(files[:max_samples//len(categories)])
        
        print(f"Found {len(self.image_files)} images for calibration")
        
        # Define preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # Add batch dimension


def collect_calibration_data(data_dir, output_file, img_size=224, max_samples=100):
    """Collect calibration data for quantization."""
    
    # Create dataset
    dataset = CalibrationDataset(data_dir, img_size, max_samples)
    
    print("Collecting calibration data...")
    
    calib_data = []
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        calib_data.append(sample.numpy())
    
    # Stack all samples
    calib_data = np.vstack(calib_data)
    print(f"Collected calibration data shape: {calib_data.shape}")
    
    # Save calibration data
    np.save(output_file, calib_data)
    print(f"Calibration data saved to {output_file}")
    
    return calib_data


def quantize_model(model_path, calib_data, output_path, target_platform="esp32s3", 
                  quant_method="per-tensor"):
    """Quantize the ONNX model using ESP-PPQ."""
    
    print(f"Quantizing model {model_path}...")
    
    # Ensure calibration data is numpy array
    if isinstance(calib_data, str):
        # If calibration data is a file path
        if os.path.exists(calib_data):
            calib_data = np.load(calib_data)
        else:
            raise FileNotFoundError(f"Calibration data file {calib_data} not found")
    
    # Quantize model using ESP-PPQ
    quantize.quantize(
        model=model_path,
        calib_data=calib_data,
        output=output_path,
        target_platform=target_platform,
        method=quant_method
    )
    
    print(f"Model quantized and saved to {output_path}")


def create_spiffs_image(model_path, output_image, model_name='/motherboard_detection.espdl', 
                       size=0x100000):
    """Create a SPIFFS image with the quantized model."""
    
    print(f"Creating SPIFFS image...")
    
    # Create SPIFFS image with the model
    write_to_spiffs.write_to_spiffs(
        model=model_path,
        spiffs=output_image,
        path=model_name,
        size=size
    )
    
    print(f"SPIFFS image created at {output_image}")
    print(f"Model is available at '{model_name}' in the SPIFFS filesystem")


def main():
    """Main function to parse arguments and run model quantization."""
    
    parser = argparse.ArgumentParser(description="ESP32-S3 Motherboard Detection Model Quantization")
    
    parser.add_argument("--onnx_model", type=str, default="models/onnx/motherboard_detector.onnx",
                       help="Path to the ONNX model to quantize")
    
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Base directory for dataset")
    
    parser.add_argument("--output_dir", type=str, default="models/espdl",
                       help="Directory to save quantized model")
    
    parser.add_argument("--img_size", type=int, default=224,
                       help="Input image size")
    
    parser.add_argument("--calib_samples", type=int, default=100,
                       help="Number of calibration samples to use")
    
    parser.add_argument("--target_platform", type=str, default="esp32s3",
                       choices=["esp32s3", "esp32s2", "esp32"],
                       help="Target ESP platform")
    
    parser.add_argument("--quant_method", type=str, default="per-tensor",
                       choices=["per-tensor", "per-channel"],
                       help="Quantization method")
    
    parser.add_argument("--create_spiffs", action="store_true",
                       help="Create SPIFFS image with the quantized model")
    
    parser.add_argument("--spiffs_size", type=str, default="0x100000",
                       help="Size of the SPIFFS image in hex (default: 1MB)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output paths
    calib_data_path = os.path.join(args.output_dir, "calib_data.npy")
    espdl_model_path = os.path.join(args.output_dir, "motherboard_detection.espdl")
    spiffs_image_path = os.path.join(args.output_dir, "spiffs.bin")
    
    # Collect calibration data
    calib_data = collect_calibration_data(
        args.data_dir, calib_data_path, 
        args.img_size, args.calib_samples
    )
    
    # Quantize model
    quantize_model(
        args.onnx_model, calib_data, espdl_model_path,
        args.target_platform, args.quant_method
    )
    
    # Create SPIFFS image if requested
    if args.create_spiffs:
        spiffs_size = int(args.spiffs_size, 16)  # Convert hex string to int
        create_spiffs_image(
            espdl_model_path, spiffs_image_path, 
            model_name='/motherboard_detection.espdl',
            size=spiffs_size
        )
        
        print("\nTo flash the SPIFFS image to your ESP32-S3, use:")
        print(f"python -m esptool.py --chip esp32s3 --port <PORT> write_flash <PARTITION_ADDR> {spiffs_image_path}")
        print("\nWhere <PORT> is your device's serial port and <PARTITION_ADDR> is the address of your SPIFFS partition.")
    
    print("\nQuantization completed successfully!")


if __name__ == "__main__":
    main()
