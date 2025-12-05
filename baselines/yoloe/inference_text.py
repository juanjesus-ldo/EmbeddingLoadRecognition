#!/usr/bin/env python3

# ./inference_text.py -if ../1064-602-T/input/images-optional/ -tp load --save_txt --save_times

import os
import argparse
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLOE
from tqdm import tqdm

def get_args():
    """
    Parses and returns command-line arguments for text-prompt-based detection.
    """
    parser = argparse.ArgumentParser(description="Run YOLOE on a folder of images using a text prompt.")
    
    parser.add_argument(
        "-if",
        "--image_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing target images to process."
    )
    parser.add_argument(
        "-tp",
        "--text_prompt",
        type=str,
        nargs='+',
        default=["load"],
        help="The text prompt(s) for the object(s) to detect. E.g., 'load' or 'person car'."
    )
    parser.add_argument(
        "--save_txt", 
        action="store_true", 
        help="If set, save a .txt file for each image with the highest confidence detected object."
    )
    parser.add_argument(
        "--save_times", 
        action="store_true", 
        help="If set, save a 'processing_times-T.txt' with performance metrics."
    )
    
    return parser.parse_args()

def main():
    """
    Main function to execute the object detection script using YOLOE with text prompts.
    """
    args = get_args()

    # --- 1. Setup and Path Validation ---
    print("Setting up model and environment...")
    
    image_folder_path = Path(args.image_folder)

    if not image_folder_path.is_dir():
        print(f"Error: The specified target image folder does not exist: {args.image_folder}")
        return

    # Create the 'detection-results' subdirectory if --save_txt is used
    if args.save_txt:
        output_txt_dir = image_folder_path / "detection-results"
        output_txt_dir.mkdir(exist_ok=True)
        print(f"Detection .txt files will be saved in: {output_txt_dir}")

    # --- 2. Model Loading and Text Prompting ---
    model_path = "yoloe-v8l-seg.pt" # Using a pre-trained model
    print(f"Loading model '{model_path}'...")
    model = YOLOE(model_path)
    print("Model loaded successfully.")

    print("Generating Text Prompt Embedding (TPE) from text...")
    text_prompts = args.text_prompt
    
    # Generate the Text Prompt Embedding (TPE) and set the classes for the model
    model.set_classes(text_prompts, model.get_text_pe(text_prompts))
    print(f"TPE generated. The model will now search for objects corresponding to the prompt(s): {text_prompts}")

    # --- 3. Image Processing Setup ---
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in image_folder_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No images found in the specified folder.")
        return
        
    # --- 4. Performance Logging Setup ---
    log_file_path = None
    if args.save_times:
        log_file_path = image_folder_path.parent / "processing_times-T.txt"
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
            results = model.predict([str(image_path)], save=False, conf=0.1, verbose=False)
            inference_end_time = time.time()
            total_model_inference_time += (inference_end_time - inference_start_time)

            file_end_time = time.time()
            total_processing_time += (file_end_time - file_start_time)

            if args.save_txt:
                output_txt_path = output_txt_dir / f"{image_path.stem}.txt"
                
                # The result is a list containing one result object
                result = results[0]
                
                # Check if any bounding boxes were detected
                if result.boxes and len(result.boxes.xyxy) > 0:
                    # Find the bounding box with the highest confidence
                    max_conf_idx = torch.argmax(result.boxes.conf).item()
                    
                    box_tensor = result.boxes.xyxy[max_conf_idx]
                    confidence = result.boxes.conf[max_conf_idx].item()
                    class_idx = int(result.boxes.cls[max_conf_idx].item())
                    
                    # Get the corresponding class name from the input prompts
                    detected_label = "load"
                    
                    left, top, right, bottom = box_tensor.tolist()
                    
                    output_str = f"{detected_label} {confidence:.6f} {left:.0f} {top:.0f} {right:.0f} {bottom:.0f}"
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
            log_message(f"Folder: {args.image_folder} | Text Prompt(s): {args.text_prompt}")
        
        for line in log_lines:
            log_message(line)
        print("--------------------------")

if __name__ == "__main__":
    main()