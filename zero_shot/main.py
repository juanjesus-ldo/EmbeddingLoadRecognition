#!/usr/bin/env python3

import os
import cv2
import argparse
import numpy as np
import torch
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoders import DINOv2, DINOv3, CAPI, RADIO
from sklearn.decomposition import PCA

# -------------------------------
# Functions for preprocessing
# -------------------------------
def preprocess_image(image_path, scale_factor):
    """
    Loads and preprocesses the image:
      - Reads the image (assumed RGB after converting from BGR).
      - Resizes it according to the indicated factor.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width, _ = image.shape
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return image_resized

# -------------------------------
# Intraclass variance function with minimum dimension constraint (MIN_BBOX x MIN_BBOX)
# -------------------------------
def min_intraclass_variance_rect_full(image, SHMIN, SHMAX, SWMIN, SWMAX, XCD=None, YCD=None):
    """
    Iterates through the image (in grayscale) evaluating all possible rectangles
    within the SHMIN-SHMAX and SWMIN-SWMAX limits. For each rectangle, it is required that:
      - the height is >= MIN_BBOX, and
      - the width is >= MIN_BBOX.
    For each candidate, the weighted variance (between pixels inside and outside)
    is calculated, and the one that minimizes said variance is returned.
    
    If XCD, YCD are provided, the rectangle is forced to include that point.
    
    Returns:
      - best_rect: (x, y, w, h) in the feature grid.
      - min_variance: value of the minimum intraclass variance.
    """
    int_img = cv2.integral(image)
    squares = np.square(image, dtype=np.float64)
    int_sq = cv2.integral(squares)

    Horig, Worig = image.shape
    total_pixels = Horig * Worig
    sum_total = int_img[-1, -1]
    sum_sq_total = int_sq[-1, -1]
    
    min_variance = float('inf')
    best_rect = None

    # Change based on the image size. 
    # The value used for each resolution is indicated in section 4.2 of the paper.
    MIN_BBOX = 7

    # It is imposed that the height and width are at least MIN_BBOX
    for h in range(max(SHMIN, MIN_BBOX), SHMAX + 1):
        for w in range(max(SWMIN, MIN_BBOX), SWMAX + 1):
            for y in range(0, Horig - h):
                for x in range(0, Worig - w):
                    if XCD is not None and YCD is not None:
                        if not (x <= XCD <= x + w and y <= YCD <= y + h):
                            continue
                    A = int_img[y, x]
                    B = int_img[y, x + w]
                    C = int_img[y + h, x]
                    D = int_img[y + h, x + w]
                    sum_inside = D + A - B - C

                    A2 = int_sq[y, x]
                    B2 = int_sq[y, x + w]
                    C2 = int_sq[y + h, x]
                    D2 = int_sq[y + h, x + w]
                    sum_sq_inside = D2 + A2 - B2 - C2

                    sum_outside = sum_total - sum_inside
                    sum_sq_outside = sum_sq_total - sum_sq_inside
                    count_inside = h * w
                    count_outside = total_pixels - count_inside

                    if count_inside <= 0 or count_outside <= 0:
                        continue

                    mean_inside = sum_inside / count_inside
                    mean_outside = sum_outside / count_outside
                    var_inside = (sum_sq_inside / count_inside) - (mean_inside ** 2)
                    var_outside = (sum_sq_outside / count_outside) - (mean_outside ** 2)

                    w1 = count_inside / total_pixels
                    w2 = count_outside / total_pixels
                    sigma2_w = w1 * var_inside + w2 * var_outside

                    if sigma2_w < min_variance:
                        min_variance = sigma2_w
                        best_rect = (x, y, w, h)
    return best_rect, min_variance

# -------------------------------
# Combined function: PCA + min_variance_intraclass
# -------------------------------
def process_and_display_image_combined(image_path, scale_factor, device, model):
    """
    Processes the image by combining:
      1. Feature extraction with the selected model.
      2. Application of PCA to obtain the 1st component (normalized and in grid form).
      3. Conversion of said grid to a grayscale image.
      4. Calculation of the optimal bounding box using intraclass variance (computation time is reported).
      5. Interactive visualization: the original image (with bounding box) and the PCA image are shown.
         The 'a' key (previous image) or 'd' key (next image) press is detected for navigation.
    """
    # Preprocess image and extract features
    start_time = time.time()
    rgb_image_numpy = preprocess_image(image_path, scale_factor)
    patch_embeddings, (h_tiles, w_tiles) = model.get_features(rgb_image_numpy)
    elapsed_time_features = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"Preprocessing and feature extraction time: {elapsed_time_features:.3f} ms")
    
    # Measure PCA computation time
    start_time = time.time()

    # Apply PCA to the patch embeddings
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(patch_embeddings)
    pca_min = pca_components.min(axis=0)
    pca_max = pca_components.max(axis=0)
    pca_norm = (pca_components - pca_min) / (pca_max - pca_min + 1e-5)
    
    # Extract the 1st component and reorganize into a grid (dim: h_tiles x w_tiles)
    pca_first = pca_norm[:, 0]
    pca_grid = pca_first.reshape(h_tiles, w_tiles)
    
    # Convert to grayscale image (0-255)
    pca_gray = (pca_grid * 255).astype(np.uint8)
    
    elapsed_time_pca = (time.time() - start_time) * 1000  # Convert to milliseconds
    print(f"PCA computation time: {elapsed_time_pca:.3f} ms")
    
    # Define limits for bounding box search in the grid
    SHMIN, SHMAX = max(1, int(0.1 * h_tiles)), h_tiles - 1
    SWMIN, SWMAX = max(1, int(0.1 * w_tiles)), w_tiles - 1
    
    # Measure bounding box computation time
    start_time = time.time()
    bbox, min_var = min_intraclass_variance_rect_full(pca_gray, SHMIN, SHMAX, SWMIN, SWMAX)
    elapsed_time_rectangle = time.time() - start_time
    print(f"Bounding box computation time: {elapsed_time_rectangle:.3f} seconds")
    
    # Prepare the original image for visualization
    orig = rgb_image_numpy / 255.0
    
    tile_size = 14  # Each patch corresponds to 14 pixels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image with bounding box
    ax1.imshow(orig)
    ax1.axis('off')
    if bbox is not None:
        x, y, w, h = bbox
        rect_orig = plt.Rectangle((x * tile_size, y * tile_size), w * tile_size, h * tile_size,
                                  edgecolor='r', facecolor='none', linewidth=2)
        ax1.add_patch(rect_orig)
        ax1.set_title(f"Original Image\n(BBox, Variance: {min_var:.2f})")
    else:
        ax1.set_title("Original Image (no bbox)")
    
    # PCA image (1st component) in grayscale
    ax2.imshow(pca_gray, cmap='gray')
    ax2.axis('off')
    ax2.set_title("PCA - 1st Component (Grayscale)")
    
    plt.suptitle(os.path.basename(image_path))
    
    # Configure keyboard navigation ('a' for previous, 'd' for next)
    navigation = {"key": None}
    def on_key(event):
        if event.key in ['a', 'd']:
            navigation["key"] = event.key
            plt.close(fig)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
    plt.show()
    return navigation["key"], bbox, min_var, elapsed_time_features, elapsed_time_pca, elapsed_time_rectangle*1000

def log_message(message):
        if args.save_times:
            with open('times_training_pca_nocode_capi_350_196.txt', 'a') as f:
                f.write(message + '\n')
        else:
            print(message)

# -------------------------------
# Main function to process a folder of images
# -------------------------------
def main(args):
    folder_path = args.folder_path
    scale_factor = args.scale_factor

    # Create the features model based on the selected model
    if args.dinov2_model:
        model = DINOv2(model_name=args.dinov2_model, half_precision=args.half_precision)
    elif args.dinov3_model:
        model = DINOv3(model_name=args.dinov3_model, half_precision=args.half_precision)
    elif args.capi_model:
        model = CAPI(model_name=args.capi_model, half_precision=args.half_precision)
    elif args.radio_model:
        model = RADIO(model_version=args.radio_model, half_precision=args.half_precision)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path))
             if f.lower().endswith(valid_exts)]
    if not files:
        print("No images found in the folder.")
        return
    
    # Interactive mode
    if args.step_by_step:
        current_index = 0
        while True:
            print(f"\nProcessing image {current_index+1} of {len(files)}: {files[current_index]}")
            key, bbox, var, _, _, _ = process_and_display_image_combined(files[current_index], scale_factor, model.device, model)
            if key == 'd':
                current_index = (current_index + 1) % len(files)
            elif key == 'a':
                current_index = (current_index - 1) % len(files)
            else:
                break
    else:
        results = []
        total_time_features, total_time_pca, total_time_rectangle = 0, 0, 0
        start_time_total = time.time()  # Measure total loop time
        for file in files:
            print(f"\nProcessing image: {file}")
            key, bbox, var, time_features, time_pca, time_rectangle = process_and_display_image_combined(file, scale_factor, model.device, model)
            results.append((file, bbox, var))
            total_time_features += time_features
            total_time_pca += time_pca
            total_time_rectangle += time_rectangle
        elapsed_time_total = (time.time() - start_time_total) * 1000  # Convert to milliseconds

        # Calculate average times
        num_files = len(files)
        avg_time_per_file = elapsed_time_total / num_files
        avg_time_features = total_time_features / num_files
        avg_time_pca = total_time_pca / num_files
        avg_time_rectangle = total_time_rectangle / num_files

        # Use log_message to save or display times
        log_message(f"\nTotal processing time: {elapsed_time_total:.3f} ms")
        log_message(f"Average time per file (Features): {avg_time_features:.3f} ms")
        log_message(f"Average time per file (PCA): {avg_time_pca:.3f} ms")
        log_message(f"Average time per file (Bounding Box): {avg_time_rectangle:.3f} ms")
        log_message(f"Average time per file: {avg_time_per_file:.3f} ms")

        # If saving to .txt is requested, the corresponding files are generated.
        if args.save_txt:
            output_dir = os.path.join(folder_path, "pca_nocode_bounding_boxes_capi_350_196")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for file, bbox, var in results:
                if bbox is not None:
                    x, y, w_box, h_box = bbox
                    # Convert grid coordinates to pixels (tile_size = 14)
                    left = x * 14
                    top = y * 14
                    right = (x + w_box) * 14
                    bottom = (y + h_box) * 14
                    base = os.path.splitext(os.path.basename(file))[0]
                    txt_file = os.path.join(output_dir, base + ".txt")
                    with open(txt_file, "w") as f:
                        f.write(f"load {var:.4f} {left} {top} {right} {bottom}\n")
                    print(f"Saved file: {txt_file}")
                    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Combination of PCA and min_variance_intraclass for bounding box extraction without code position")
    parser.add_argument('-fp', "--folder_path", type=str, required=True, help="Path to the folder with images to process")
    parser.add_argument('-sf', "--scale_factor", type=float, default=1.0, help="Scale factor for the image")
    parser.add_argument('--step_by_step', action='store_true', help="Process image by image interactively")
    parser.add_argument('--save_txt', action='store_true', help="Generate .txt files with detection for each image (batch mode only)")
    parser.add_argument('--save_times', action='store_true', help="Save processing times in times_training_pca_nocode.txt")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dinov2_model', choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
             'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'],
        help='DINOv2 model to use')
    group.add_argument('--dinov3_model', choices=['dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 'dinov3_vitl16',
             'dinov3_vith16plus'],
        help='DINOv3 model to use')
    group.add_argument('--capi_model', choices=['capi_vitl14_p205', 'capi_vitl14_lvd', 'capi_vitl14_in22k', 'capi_vitl14_in1k'],
        help='CAPI model to use')
    group.add_argument('--radio_model', choices=['radio_v2.5-g', 'radio_v2.5-h', 'radio_v2.5-l', 'radio_v2.5-b', 'e-radio_v2'],
        help='RADIO model to use')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision for the model.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # If interactive mode is not used, the matplotlib backend is changed to avoid interactive windows.
    if not args.step_by_step:
        import matplotlib
        matplotlib.use('Agg')
        plt.show = lambda: plt.close('all')
    
    main(args)