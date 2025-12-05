#!/usr/bin/env python3

# ./inference.py -if ./1064-602/input/images-optional/ -p "load." --save_txt --save_times

import os
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm

def get_args():
    """
    Parses and returns command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Florence-2 on a folder of images for object detection.")
    
    parser.add_argument(
        "-if",
        "--image_folder", 
        type=str, 
        required=True, 
        help="Path to the folder containing images to process."
    )
    parser.add_argument(
        "-p",
        "--prompt", 
        type=str, 
        required=True, 
        help="Text prompt for phrase grounding (e.g., 'a cat' or 'a remote control')."
    )
    parser.add_argument(
        "--save_txt", 
        action="store_true", 
        help="If set, save a .txt file for each image with the first detected object."
    )
    parser.add_argument(
        "--save_times", 
        action="store_true", 
        help="If set, save a 'processing_times.txt' with performance metrics."
    )
    
    return parser.parse_args()

def main():
    """
    Main function to execute the object detection script using Florence-2.
    """
    args = get_args()

    # --- 1. Setup and Model Loading ---
    print("Setting up model and environment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "microsoft/Florence-2-base-ft"
    
    image_folder_path = Path(args.image_folder)
    if not image_folder_path.is_dir():
        print(f"Error: The specified image folder does not exist: {args.image_folder}")
        return

    # Create the 'detection-results' subdirectory if --save_txt is used
    if args.save_txt:
        output_txt_dir = image_folder_path / "detection-results"
        output_txt_dir.mkdir(exist_ok=True)
        print(f"Detection .txt files will be saved in: {output_txt_dir}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    print(f"Model '{model_id}' loaded successfully on device: {device}")

    # --- 2. Image Processing Setup ---
    # The task prompt for Florence-2 phrase grounding
    florence_task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    caption = args.prompt

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = [f for f in image_folder_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No images found in the specified folder.")
        return
        
    # --- 3. Performance Logging Setup ---
    log_file_path = None
    if args.save_times:
        user_path = Path(args.image_folder)
        
        if user_path.is_absolute():
            root_dir = user_path.parent
        else:
            path_parts = user_path.parts
            root_dir = Path(path_parts[0]) if path_parts else Path('.')

        log_file_path = root_dir / "processing_times.txt"
        
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Performance metrics will be saved to: {log_file_path.resolve()}")


    # --- 4. Main Processing Loop ---
    total_processing_time = 0
    total_model_inference_time = 0
    num_files = len(image_files)
    
    master_start_time = time.time()

    print(f"\nStarting processing for {num_files} images...")
    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            file_start_time = time.time()
            
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs for Florence-2
            inputs = processor(text=f"{florence_task_prompt} {caption}", images=image, return_tensors="pt").to(device, torch_dtype)

            # Generate model output
            inference_start_time = time.time()
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
            inference_end_time = time.time()
            total_model_inference_time += (inference_end_time - inference_start_time)

            # Decode and post-process the results
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            results = processor.post_process_generation(
                generated_text,
                task=florence_task_prompt,
                image_size=(image.width, image.height)
            )
            
            file_end_time = time.time()
            total_processing_time += (file_end_time - file_start_time)

            if args.save_txt:
                output_txt_path = output_txt_dir / f"{image_path.stem}.txt"
                
                # Extract bounding boxes for the specified task
                detections = results.get(florence_task_prompt, {})
                bboxes = detections.get('bboxes', [])
                
                if bboxes:
                    # Florence-2 does not provide a confidence score.
                    # We will save the first detected bounding box with a placeholder score of 1.0.
                    first_box = bboxes[0]
                    placeholder_score = 1.0
                    left, top, right, bottom = first_box
                    
                    # Force label to "load" as in the original script
                    output_label = "load"
                    
                    output_str = f"{output_label} {placeholder_score:.6f} {left:.0f} {top:.0f} {right:.0f} {bottom:.0f}"
                    with open(output_txt_path, 'w') as f:
                        f.write(output_str)
                else:
                    # Create an empty file if no objects are detected
                    output_txt_path.touch()
        except Exception as e:
            print(f"Could not process file {image_path.name}: {e}")

    master_end_time = time.time()
    elapsed_time_total_s = master_end_time - master_start_time

    # --- 5. Final Performance Reporting ---
    if num_files > 0:
        avg_time_model_ms = (total_model_inference_time / num_files) * 1000
        avg_time_per_file_ms = (total_processing_time / num_files) * 1000
        elapsed_time_total_ms = elapsed_time_total_s * 1000

        log_lines = [
            f"Total processing time: {elapsed_time_total_ms:.3f} ms",
            f"Average time per file (florence-2 generation): {avg_time_model_ms:.3f} ms",
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
            log_message(f"Folder: {args.image_folder} | Prompt: '{args.prompt}'")
        
        for line in log_lines:
            log_message(line)
        print("--------------------------")

if __name__ == "__main__":
    main()