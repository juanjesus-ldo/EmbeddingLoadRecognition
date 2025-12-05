#!/usr/bin/env python

import argparse
import os
import json
import time

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encoders import DINOv2, DINOv3, CAPI, RADIO
from src.eval_classification import calculate_recall_precision, build_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import cupy as cp

BATCH_SIZE = 60000

MODEL_COLORS = {
    'class_1': (101, 159, 252),
    'class_2': (204, 158, 120),
    'class_3': (192, 192, 192),
    'class_4': (252, 131, 16),
    'class_5': (125, 34, 191),
    'class_6': (250, 110, 110),
    'class_7': (255, 0, 0),
    'class_8': (235, 247, 5),
    'class_9': (166, 237, 229),
    'class_10': (124, 214, 124),
    'class_11': (0, 128, 0),
    'background': (77, 77, 77),
}

# Parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_folder', type=str, help='''Path to the folder with the images to process.
                         The folder must contain one subfolder per class.
                         In addition, each of them must contain the subfolders:
                         - images : Images in .jpg format
                         - masks : Masks in .png format (Used only to create the models)
                         ''')
    parser.add_argument('models_config', type=str, help='''JSON file with the images to use as a model.
                         This has two properties:
                         - models_names: An object where each key is a category that contains a list of file names.
                         - bg_names: A list of file names.''')
    parser.add_argument('output_folder', type=str, help='Path to the folder where the results will be saved.')
    parser.add_argument('--num_models', type=int, default=1, help='Number of images to use as a model per class.')
    parser.add_argument('--num_bg_images', type=int, default=0, help='Number of background images to use.')
    parser.add_argument('--bg_folder', type=str, default=None, help='Path to the folder with background images.')
    parser.add_argument('--pca_components', type=int, default=30, help='Number of components to use for PCA. -1 means no PCA is used.')
    parser.add_argument('--n_neighbors', type=int, default=7, help='Number of neighbors to use for the KNN model.')
    parser.add_argument('--postprocess', type=int, default=5, help='Kernel size for post-processing convolution. 1 means no convolution is used.')
    parser.add_argument('--show', action='store_true', help='Show the predictions on screen.')
    parser.add_argument('--save_show', type=str, default=None, help='Saves what is shown with --show. This option is also necessary.')
    parser.add_argument('--cupy', action='store_true', help='Use CuPy to speed up the KNN model.')
    parser.add_argument('--embeddings_folder', type=str, default=None, help='If specified, embeddings will be searched for in this folder before computing them. If not found, they will be computed and saved in this folder.')
    parser.add_argument('--no_encoder_model', action='store_true', help='If you are sure that all embeddings have already been computed, you can use this option to avoid loading the encoder model.')
    parser.add_argument('--show_times', action='store_true', help='Display the time it takes for each image to be classified on the console.')

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

# Load the JSON file with the images names to use as models. Returns a 
# list with the names of the classes and two dictionaries with the 
# names of the images to use as models and as background
def load_models_config(models_config):
    with open(models_config, 'r') as f:
        data = json.load(f)

    models_names = data['models_names']
    bg_names = data['bg_names']

    return list(models_names.keys()), models_names, bg_names

# Get the features from an image. If embeddings_folder is not None, it will
# look for the embeddings in that folder. If not, it will compute the features
# using the features model and save them in the embeddings folder
def get_features_model(features_model, image, embeddings_folder=None, image_name=None):
    if embeddings_folder is None:
        return features_model.get_features(image)
    else:
        # Search for the embeddings in the folder
        image_name = image_name.replace('.jpg', '.npz')
        embeddings_path = os.path.join(embeddings_folder, image_name)
        if os.path.exists(embeddings_path):
            # Load the embeddings from the file
            data = np.load(embeddings_path)
            features = data['features']
            grid_size = tuple(data['grid_size'])
            return features, grid_size
        else:
            # Compute the features and save them in the folder
            features, grid_size = features_model.get_features(image)
            np.savez(embeddings_path, features=features, grid_size=grid_size)
            return features, grid_size

# Get the features from a (masked) image. Returns the feature vectors and 
# a list of the same size with the respective classes
def get_mb_features(image, mask, features_model, class_name, classes, embeddings_folder=None, image_name=None):
    # Compute the features vector
    features, grid_size = get_features_model(features_model, image, embeddings_folder=embeddings_folder, image_name=image_name)
    if class_name == 'background':
        # If the class is background, we use the whole image
        features_classes = np.full((grid_size[0], grid_size[1]), len(classes)).astype(np.int32)
    else:
        # Mask the features
        resized_mask = Image.fromarray(mask)
        resized_mask = np.asarray(resized_mask.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)).astype(bool)
        # Use the patches ouf of the mask as background
        features_classes = np.where(resized_mask, classes.index(class_name), len(classes)).astype(np.int32)
    return features, features_classes

# Load the images and masks and extract features for creating the memory bank
# Memory bank consists in two numpy arrays: one with the features and another with the classes associated to each feature
def build_memorybank(input_folder, models_names, num_models, bg_folder, bg_names, num_bg_images, classes, features_model, pca_components, embeddings_folder=None):
    mb_features, mb_classes = [], []
    # Load the images and its maks to use as models
    for clss in classes:
        for name in models_names[clss][:num_models]:
            img_path = os.path.join(input_folder, clss, 'images', name)
            mask_path = os.path.join(input_folder, clss, 'masks', name.replace('jpg', 'png'))
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8)
            # Get the features
            features, features_classes = get_mb_features(image, mask, features_model, clss, classes, embeddings_folder=embeddings_folder, image_name=name)
            mb_features.append(features)
            mb_classes.extend(features_classes)

    # Same for the background images
    if num_bg_images > 0:
        for name in bg_names[:num_bg_images]:
            image = cv2.cvtColor(cv2.imread(os.path.join(bg_folder, name)), cv2.COLOR_BGR2RGB)
            features, features_classes = get_mb_features(image, mask, features_model, 'background', classes, embeddings_folder=embeddings_folder, image_name=name)
            mb_features.append(features)
            mb_classes.extend(features_classes)

    # Final concatenation of the features and classes
    mb_features = np.concatenate(mb_features, axis=0)
    mb_classes = np.concatenate(mb_classes, axis=0)

    # If PCA is used, apply it to the features
    pca_model = None
    if pca_components > 0:
        components = min(pca_components, mb_features.shape[1], mb_features.shape[0])
        pca_model = PCA(n_components = components)
        mb_features = pca_model.fit_transform(mb_features)

    return mb_features, mb_classes, pca_model

# Create the knn model using the memory bank
def create_knn_model(n_neighbors, mb_features, use_cupy, metric='cosine'):
    if use_cupy:
        mb_features = cp.asarray(mb_features)
        # Normalize the features for cosine distance
        mb_features = mb_features / cp.linalg.norm(mb_features, axis=1, keepdims=True)
        return {    # Returns a dict simulating the knn model
            'features': mb_features,
            'n_neighbors': n_neighbors,
            'metric': metric
        }
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(mb_features)
    return knn

# Search for the nearest neighbors using the knn model
# If use_cupy is True, it uses CuPy to accelerate the search
def knn_search(model, query_vectors, use_cupy, batch_size=BATCH_SIZE):
    if use_cupy:
        mb_features = model['features']
        n_neighbors = model['n_neighbors']
        # Convert query vectors to GPU and normalize
        query_vectors = cp.asarray(query_vectors)
        query_vectors = query_vectors / cp.linalg.norm(query_vectors, axis=1, keepdims=True)
        # Preallocate top-k storage
        n_queries = query_vectors.shape[0]
        topk_similarities = cp.full((n_queries, n_neighbors), -cp.inf, dtype=cp.float32)
        topk_indices = cp.full((n_queries, n_neighbors), -1, dtype=cp.int32)
        mb_features_gpu = cp.asarray(mb_features)  # Move all features once if possible

        batch_size = min(batch_size, mb_features_gpu.shape[0])
        for i in range(0, mb_features_gpu.shape[0], batch_size):
            chunk = mb_features_gpu[i:i + batch_size]
            chunk = chunk / cp.linalg.norm(chunk, axis=1, keepdims=True)

            # Compute similarities
            sims = cp.matmul(query_vectors, chunk.T)
            # For each query, find the local top-k in this chunk
            local_topk = cp.argsort(-sims, axis=1)[:, :n_neighbors]
            local_sims = cp.take_along_axis(sims, local_topk, axis=1)
            global_indices = local_topk + i
            # Combine the local top-k with the global top-k
            combined_sims = cp.concatenate([topk_similarities, local_sims], axis=1)
            combined_indices = cp.concatenate([topk_indices, global_indices], axis=1)
            # Sort and keep the top-k
            new_topk = cp.argsort(-combined_sims, axis=1)[:, :n_neighbors]
            topk_similarities = cp.take_along_axis(combined_sims, new_topk, axis=1)
            topk_indices = cp.take_along_axis(combined_indices, new_topk, axis=1)

            # Free memory
            del chunk, sims, local_topk, local_sims, global_indices
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        # Final distances are computed as 1 - cosine similarity
        topk_distances = 1 - topk_similarities
        return topk_distances.get(), topk_indices.get()

    else:
        distances, indices = model.kneighbors(query_vectors)
        return distances, indices

# Convolution on an prediction image ignoring the background patches
def conv_no_background(img, bg_id, ksize=3):
    pad = ksize // 2
    padded = np.pad(img, pad, mode='edge')
    output = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == bg_id: # Skip background pixels
                continue
            patch = padded[i:i+ksize, j:j+ksize]
            patch = patch.flatten()
            # Ignore the background patches for computing the median
            non_zero_patch = patch[patch != bg_id]
            if len(non_zero_patch) > 0:
                values, counts = np.unique(non_zero_patch, return_counts=True)
                output[i, j] = values[np.argmax(counts)]
    return output.astype(img.dtype)

# Returns the predicted classes for every patch of the image
def classify_image(img_path, features_model, pca_model, knn_model, mb_classes, bg_id, postprocess, use_cupy, embeddings_folder=None):
    # 1. Load the image and compute the features
    t0 = time.time()
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_name = os.path.basename(img_path)
    features, grid_size = get_features_model(features_model, image, embeddings_folder=embeddings_folder, image_name=image_name)
    time_features = time.time() - t0

    # 2. Project every feature into the PCA space (if requested)
    vectors_for_knn = features
    if pca_model is not None:
        vectors_for_knn = pca_model.transform(features)
    
    # 3. For every new vector, find the nearest neighbor
    t0 = time.time()
    distances, matches = knn_search(knn_model, vectors_for_knn, use_cupy=use_cupy)
    time_knn = time.time() - t0

    pred_patches = []
    # 4. Weighted voting using inverse distances
    epsilon = 1e-5  # to avoid division by zero
    for dists, idxs in zip(distances, matches):
        class_weights = {}
        for dist, idx in zip(dists, idxs):
            label = mb_classes[idx]
            weight = 1.0 / (dist + epsilon)
            class_weights[label] = class_weights.get(label, 0) + weight
        # Choose the class with the highest total weight
        pred_patches.append(max(class_weights, key=class_weights.get))

    pred_patches = np.array(pred_patches, dtype='uint8')
    pred_patches = pred_patches.reshape(grid_size[0], grid_size[1])

    # 5. Take the biggest connected component
    mask = (pred_patches != bg_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    x1, y1, w, h = cv2.boundingRect(biggest_contour)
    x2, y2 = x1 + w, y1 + h
    pred_patches_con = np.full_like(pred_patches, bg_id)
    pred_patches_con[y1:y2, x1:x2] = pred_patches[y1:y2, x1:x2]

    # 6. Post-processing convolution
    if postprocess > 1:
        pred_patches_con = conv_no_background(pred_patches_con, bg_id, ksize=postprocess)

    # 7. Compute the global image prediction
    no_bg = pred_patches_con[pred_patches_con != bg_id]
    values, counts = np.unique(no_bg, return_counts=True)
    id_max = np.argmax(counts)
    pred_class_id = values[id_max]
    percentages = {int(v): count / len(no_bg) for v, count in zip(values, counts)}

    # 8. Upscale the bounding box to the original image size
    scale_x = image.shape[1] / grid_size[1]
    scale_y = image.shape[0] / grid_size[0]
    x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
    x2, y2 = int(x2 * scale_x), int(y2 * scale_y)

    return { 'pred_patches': pred_patches_con, 'percentages': percentages, 'pred_class_id': pred_class_id,
        'time_features': time_features, 'time_knn': time_knn, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2 } 

### MAIN FUNCTION ###
def main():
    args = parse_arguments()
    input_folder = args.input_folder
    if not os.path.exists(input_folder):
        print(f'[!]: The input folder {input_folder} does not exist.')
        return
    if args.num_bg_images > 0 and not args.bg_folder:
        print('If --num_bg_images > 0, --bg_folder must also be specified')
        return
    embeddings_folder = args.embeddings_folder
    if embeddings_folder is not None and not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    save_show_folder = args.save_show
    if save_show_folder is not None and not os.path.exists(save_show_folder):
        os.makedirs(save_show_folder)
    
    # Load the JSON file with the images names to use as models
    classes, models_names, bg_names = load_models_config(args.models_config)
    print(f'[+]: Using {len(classes)} classes, {args.num_models} images per class, and {args.num_bg_images} background images.')

    # Create the features model based on the selected model
    if args.no_encoder_model:
        features_model = None
    elif args.dinov2_model:
        features_model = DINOv2(model_name=args.dinov2_model, half_precision=args.half_precision)
    elif args.dinov3_model:
        features_model = DINOv3(model_name=args.dinov3_model, half_precision=args.half_precision)
    elif args.capi_model:
        features_model = CAPI(model_name=args.capi_model, half_precision=args.half_precision)
    elif args.radio_model:
        features_model = RADIO(model_version=args.radio_model, half_precision=args.half_precision)

    # Build the memory bank
    t0 = time.time()
    mb_features, mb_classes, pca_model = build_memorybank(input_folder, models_names, args.num_models, args.bg_folder, bg_names, args.num_bg_images, classes, features_model, args.pca_components, embeddings_folder=embeddings_folder)
    print(f'[+]: Memory bank created with {mb_features.shape[0]} features of size {mb_features.shape[1]} in {time.time()-t0:.2f} seconds')
    
    # Create the knn model
    use_cupy = args.cupy
    knn_model = create_knn_model(args.n_neighbors, mb_features, use_cupy)

    # Classify every image in the dataset
    output_folder = args.output_folder
    postprocess = args.postprocess
    cartucho_folder = os.path.join(output_folder, 'bboxes_cartucho')
    if not os.path.exists(cartucho_folder):
        os.makedirs(cartucho_folder)
    predictions_log, classification_log, percentages_log = [], [], []
    predictions_log.append('NAME REAL_CLASS PREDICTED_CLASS PERCENTAGE LEFT_BBOX TOP_BBOX RIGHT_BBOX BOTTOM_BBOX TIME_FEATURES TIME_KNN TIME_TOTAL')
    y_true, y_pred = [], []
    feat_times, knn_times, class_time = [], [], []
    for class_name in classes:
        folder = os.path.join(input_folder, class_name, 'images')
        per_tp, per_fn = [], []
        for name in os.listdir(folder):
            # Skip an image if its selected as a model, continue
            if name in models_names[class_name][:args.num_models]:
                continue

            # Classify the image
            t0 = time.time()
            prediction = classify_image(os.path.join(folder, name), features_model, pca_model, knn_model, mb_classes, len(classes), postprocess, use_cupy, embeddings_folder=embeddings_folder)
            time_classify = time.time() - t0
            
            # Compute and save the metrics
            pred_patches = prediction['pred_patches']
            pred_class_id = prediction['pred_class_id']
            percentage = max(prediction['percentages'].values())
            time_features = prediction['time_features']
            time_knn = prediction['time_knn']
            x1, y1, x2, y2 = prediction['x1'], prediction['y1'], prediction['x2'], prediction['y2']
            pred_class = classes[pred_class_id]
            predictions_log.append(f'{name} {class_name} {pred_class} {percentage} {x1} {y1} {x2} {y2} {time_features:.4f} {time_knn:.4f} {time_classify:.4f}')
            percentages_log.append(f"{name} " + ' '.join([f"{classes[cls]} {pct:.2f}" for cls, pct in prediction['percentages'].items()]))
            y_true.append(class_name)
            y_pred.append(pred_class)
            feat_times.append(time_features)
            knn_times.append(time_knn)
            class_time.append(time_classify)
            if class_name == pred_class:
                per_tp.append(percentage)
            else:
                per_fn.append(percentage)

            # Save the predictions (to compute the object detection metrics later)
            # Every detectetion is saved in a 'txt' file with the name of the image and
            # the format: <class_name> <confidence> <left> <top> <right> <bottom>
            with open(os.path.join(cartucho_folder, name.replace('jpg', 'txt')), 'w') as f:
                f.write(f'{pred_class} {percentage} {x1} {y1} {x2} {y2}\n')

            # Show times or the image with the prediction if requested
            if args.show_times:
                # Features + knn included in classify
                print(f'> Times: features={time_features:.4f}, knn={time_knn:.4f}, classification={time_classify:.4f}')
            if args.show:
                print(f'{name} {class_name} {pred_class} {percentage} {x1} {y1} {x2} {y2} {time_features:.4f} {time_knn:.4f} {time_classify:.4f}')
                lut = np.zeros((len(classes)+1, 3), dtype=np.uint8)
                for i, clss in enumerate(classes+['background']):
                    lut[i] = MODEL_COLORS[clss]
                patch_img = lut[pred_patches]
                patch_img = cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR)
                img = cv2.imread(os.path.join(folder, name))
                patch_img = cv2.resize(patch_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(img, (x1, y1), (x2, y2), MODEL_COLORS[pred_class][::-1], 2)
                cv2.rectangle(patch_img, (x1, y1), (x2, y2), MODEL_COLORS[pred_class][::-1], 2)
                final_show = np.hstack((img, patch_img))
                cv2.imshow('Prediction', final_show)
                if save_show_folder is not None:
                    cv2.imwrite(os.path.join(save_show_folder, name.replace('jpg', 'png')), final_show)
                key = cv2.waitKey(20)   # show for 20 ms
                if key == 32:   # space key
                    cv2.waitKey(0)
                elif key == 27:  # esc key
                    cv2.destroyAllWindows()
                    return
                
        tp, fn = len(per_tp), len(per_fn)
        per_tp, per_fn = np.array(per_tp), np.array(per_fn)
        classification_log.append(f'Total of {class_name}: {tp+fn} images')
        if tp > 0:
            classification_log.append(f'TP: {tp}, avg_score: {np.mean(per_tp)}, median_score: {np.median(per_tp)}')
        if fn > 0:
            classification_log.append(f'FN: {fn}, avg_score: {np.mean(per_fn)}, median_score: {np.median(per_fn)}')

    # Save the predictions and percentages log
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    predictions_path = os.path.join(output_folder, 'predictions.txt')
    with open(predictions_path, 'w') as f:
        for line in predictions_log:
            f.write(line + '\n')
    print(f'[+]: Predictions saved to {predictions_path}')
    percentages_path = os.path.join(output_folder, 'percentages.txt')
    with open(percentages_path, 'w') as f:
        for line in percentages_log:
            f.write(line + '\n')
    print(f'[+]: Percentages saved to {percentages_path}')

    # Compute the classification metrics and save them
    conf_mat_path = os.path.join(output_folder, 'confusion_matrix.png')
    conf_mat = build_confusion_matrix(y_true, y_pred, cmap='Blues', save_path=conf_mat_path)
    classification_log.append('-'*100+'\nConfusion Matrix:\n'+conf_mat+'\n'+'-'*100)
    labels, recall, precision, f1, recall_avg, precision_avg, f1_avg = calculate_recall_precision(y_true, y_pred)
    classification_log.append(f'Scores for {labels}:')
    classification_log.append(f'Recall: {recall}')
    classification_log.append(f'Precision: {precision}')
    classification_log.append(f'F1: {f1}')
    classification_log.append(f'Average Recall: {recall_avg}')
    classification_log.append(f'Average Precision: {precision_avg}')
    classification_log.append(f'Average F1: {f1_avg}')
    classification_log.append('-'*100)
    classification_log.append(f'Average feature extraction times: {np.mean(feat_times):.4f} seconds')
    classification_log.append(f'Average KNN search times: {np.mean(knn_times):.4f} seconds')
    classification_log.append(f'Average total classification times: {np.mean(class_time):.4f} seconds')
    cv2.destroyAllWindows()
    classification_path = os.path.join(output_folder, 'classification.txt')
    with open(classification_path, 'w') as f:
        for line in classification_log:
            f.write(line + '\n')
    print(f'[+]: Classification saved to {classification_path}')

if __name__ == '__main__':
    main()
