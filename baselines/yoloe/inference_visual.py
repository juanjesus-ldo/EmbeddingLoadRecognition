#!/usr/bin/env python3

""" ./inference_visual.py -if ../1064-602/input/images-optional/ -si ../1064-602/input/images-optional/proof_1_504_1726847916713_906_546_273_525_267_527_246_548_252.jpg\
  --bbox 212 29 608 1000 --save_txt --save_times 
"""

import os
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from tqdm import tqdm

def get_args():
    """
    Parses and returns command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run YOLOE on a folder of images using a visual prompt.")
    
    parser.add_argument(
        "-if",
        "--image_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing target images to process."
    )
    parser.add_argument(
        "-si",
        "--source_image",
        type=str,
        required=True,
        help="Path to the source image containing the object for the visual prompt."
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=('X1', 'Y1', 'X2', 'Y2'),
        help="The bounding box of the object in the source image, format: x1 y1 x2 y2."
    )
    parser.add_argument(
        "--save_txt", 
        action="store_true", 
        help="If set, save a .txt file for each image with the first detected object."
    )
    parser.add_argument(
        "--save_times", 
        action="store_true", 
        help="If set, save a 'processing_times-V.txt' with performance metrics."
    )
    
    return parser.parse_args()

def main():
    """
    Main function to execute the object detection script using YOLOE with visual prompts.
    """
    args = get_args()

    # --- 1. Setup and Path Validation ---
    print("Setting up model and environment...")
    
    image_folder_path = Path(args.image_folder)
    source_image_path = Path(args.source_image)

    if not image_folder_path.is_dir():
        print(f"Error: The specified target image folder does not exist: {args.image_folder}")
        return
    if not source_image_path.is_file():
        print(f"Error: The specified source image does not exist: {args.source_image}")
        return

    # Create the 'detection-results' subdirectory if --save_txt is used
    if args.save_txt:
        output_txt_dir = image_folder_path / "detection-results"
        output_txt_dir.mkdir(exist_ok=True)
        print(f"Detection .txt files will be saved in: {output_txt_dir}")

    # --- 2. Model Loading and Visual Prompting ---
    model_path = "yoloe-v8l-seg.pt" # Using a pre-trained model
    print(f"Loading model '{model_path}'...")
    model = YOLOE(model_path)
    print("Model loaded successfully.")

    print("Generating Visual Prompt Embedding (VPE) from source image...")
    # Construct the visual prompt from command-line arguments
    visuals = {
        "bboxes": [np.array([args.bbox])],
        "cls": [np.array([0])] # Placeholder class 0
    }
    
    # Generate the Visual Prompt Embedding (VPE)
    model.predict(str(source_image_path), prompts=visuals, predictor=YOLOEVPSegPredictor, return_vpe=True)
    
    # Set the class for detection based on the generated VPE
    output_label = "load" # Hardcoded label as in the original script
    model.set_classes([output_label], model.predictor.vpe)
    
    # Remove the VPPredictor to switch to standard inference mode
    model.predictor = None 
    print(f"VPE generated. The model will now search for objects similar to the one in the prompt with label '{output_label}'.")

    # --- 3. Image Processing Setup ---
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in image_folder_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No images found in the specified folder.")
        return
        
    # --- 4. Performance Logging Setup ---
    log_file_path = None
    if args.save_times:
        user_path = Path(args.image_folder)
        
        if user_path.is_absolute():
            root_dir = user_path.parent
        else:
            path_parts = user_path.parts
            root_dir = Path(path_parts[0]) if path_parts else Path('.')

        log_file_path = root_dir / "processing_times-V.txt"
        
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Performance metrics will be saved to: {log_file_path.resolve()}")


    # --- 5. Main Processing Loop ---
    total_processing_time = 0
    total_model_inference_time = 0
    num_files = len(image_files)
    
    master_start_time = time.time()

    print(f"\nStarting processing for {num_files} target images...")
    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            file_start_time = time.time()
            
            # Generate model output
            inference_start_time = time.time()
            # YOLOE can take a list, but we process one-by-one for accurate timing and file handling
            results = model.predict([str(image_path)], save=False, conf=0.1)
            inference_end_time = time.time()
            total_model_inference_time += (inference_end_time - inference_start_time)

            file_end_time = time.time()
            total_processing_time += (file_end_time - file_start_time)

            if args.save_txt:
                output_txt_path = output_txt_dir / f"{image_path.stem}.txt"
                
                result = results[0]
                
                # Check if any bounding boxes were detected
                if result.boxes and len(result.boxes.xyxy) > 0:
                    # Find the bounding box with the highest confidence
                    max_conf_idx = torch.argmax(result.boxes.conf).item()
                    box_tensor = result.boxes.xyxy[max_conf_idx]
                    confidence = result.boxes.conf[max_conf_idx].item()
                    left, top, right, bottom = box_tensor.tolist()
                    
                    output_str = f"{output_label} {confidence:.6f} {left:.0f} {top:.0f} {right:.0f} {bottom:.0f}"
                    with open(output_txt_path, 'w') as f:
                        f.write(output_str)
                else:
                    # Create an empty file if no objects are detected
                    output_txt_path.touch()
        except Exception as e:
            print(f"Could not process file {image_path.name}: {e}")

    master_end_time = time.time()
    elapsed_time_total_s = master_end_time - master_start_time

    # --- 6. Final Performance Reporting ---
    if num_files > 0:
        avg_time_model_ms = (total_model_inference_time / num_files) * 1000
        avg_time_per_file_ms = (total_processing_time / num_files) * 1000
        elapsed_time_total_ms = elapsed_time_total_s * 1000

        log_lines = [
            f"Total processing time: {elapsed_time_total_ms:.3f} ms",
            f"Average time per file (YOLOE prediction): {avg_time_model_ms:.3f} ms",
            f"Average time per file (total): {avg_time_per_file_ms:.3f} ms"
        ]
        
        def log_message(message):
            print(message)
            if args.save_times:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(message + '\n')
        
        print("\n--- Performance Summary ---")
        if args.save_times:
            log_message(f"\n--- Log Entry: {time.ctime()} ---")
            log_message(f"Folder: {args.image_folder} | Source Image: '{args.source_image}' | BBox: {args.bbox}")
        
        for line in log_lines:
            log_message(line)
        print("--------------------------")

if __name__ == "__main__":
    main()