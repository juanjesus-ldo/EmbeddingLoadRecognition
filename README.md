# Few-Shot and Zero-Shot Object Detection for Industrial Lateral Loads

> **WACV 2026 Submission**  
> This repository contains the official implementation of the paper **"Few-Shot and Zero-Shot Object Detection for Industrial Lateral Loads"**.

---

<div align="center">
<b>Overview of our few-shot approach for pallet load recognition.</b>

![few_shot_pipeline](./media/pipeline.jpg)

</div>

---

## 📋 Table of Contents

- [Few-Shot and Zero-Shot Object Detection for Industrial Lateral Loads](#few-shot-and-zero-shot-object-detection-for-industrial-lateral-loads)
  - [📋 Table of Contents](#-table-of-contents)
  - [📖 Introduction](#-introduction)
  - [💾 Dataset](#-dataset)
  - [🛠️ Installation](#️-installation)
  - [🚀 Usage](#-usage)
    - [Few-Shot Method](#few-shot-method)
    - [Zero-Shot Method](#zero-shot-method)
    - [Baselines](#baselines)
      - [Florence-2 (Zero-Shot)](#florence-2-zero-shot)
      - [YOLOE (Visual Prompt)](#yoloe-visual-prompt)
      - [YOLOE (Text Prompt)](#yoloe-text-prompt)
  - [📊 Evaluation](#-evaluation)
  - [📜 Citation](#-citation)
  - [📄 License](#-license)

## 📖 Introduction

This project presents a novel approach for object detection in industrial environments, specifically focusing on lateral loads. We propose **Few-Shot** and **Zero-Shot** methodologies that leverage state-of-the-art foundation models (DINOv2, DINOv3, CAPI, RADIO) to achieve high performance with minimal or no annotated data.

Our approach addresses the challenges of:
- **Data Scarcity**: Reducing the need for large labeled datasets.
- **Adaptability**: Quickly adapting to new object classes (e.g., different types of loads).
- **Robustness**: Handling complex industrial backgrounds and occlusions.

## 💾 Dataset

The experiments in this paper were conducted using the **IndustrialLateralLoads** dataset.

- **Hugging Face Dataset**: [jjldo21/IndustrialLateralLoads](https://huggingface.co/datasets/jjldo21/IndustrialLateralLoads)

This dataset contains images of various industrial loads in realistic warehouse settings, annotated with bounding boxes.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/juanjesus-ldo/EmbeddingLoadRecognition.git
    cd EmbeddingLoadRecognition
    ```

2.  **Install dependencies:**
    We recommend using a virtual environment (e.g., conda or venv).
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

The repository is organized into three main sections: `few_shot`, `zero_shot`, and `baselines`.

### Few-Shot Method

The Few-Shot approach uses a small set of support images (shots) to detect objects in query images.

**Script:** `few_shot/main.py`

**Arguments:**
- `input_folder`: Path to the folder with images to process (organized by class).
- `models_config`: JSON file defining the support images (models) and background images.
- `output_folder`: Path to save results.
- `--num_models`: Number of shots per class (default: 1).
- `--dinov2_model`, `--dinov3_model`, `--capi_model`, `--radio_model`: **(Required)** Select the encoder architecture.
- `--half_precision`: Use half precision (fp16) for the model.

**Example:**
```bash
python3 few_shot/main.py \
    /path/to/dataset \
    /path/to/config.json \
    /path/to/output \
    --num_models 5 \
    --dinov2_model dinov2_vitb14
```

### Zero-Shot Method

The Zero-Shot approach detects objects without any specific training examples for the target class, relying on generic feature extraction and intraclass variance minimization.

**Script:** `zero_shot/main.py`

**Arguments:**
- `-fp`, `--folder_path`: Path to the folder with images to process.
- `--dinov2_model`, `--dinov3_model`, `--capi_model`, `--radio_model`: **(Required)** Select the encoder architecture.
- `--save_txt`: Generate .txt files with detections.
- `--step_by_step`: Run interactively.

**Example:**
```bash
python3 zero_shot/main.py \
    -fp /path/to/images \
    --capi_model capi_vitl14_p205 \
    --save_txt
```

### Baselines

We compare our methods against strong baselines like **Florence-2** and **YOLOE**.

#### Florence-2 (Zero-Shot)
```bash
python3 baselines/florence2/inference.py \
    -if /path/to/images \
    -p "load" \
    --save_txt
```

#### YOLOE (Visual Prompt)
```bash
python3 baselines/yoloe/inference_visual.py \
    -if /path/to/images \
    -si /path/to/source_image.jpg \
    --bbox x1 y1 x2 y2 \
    --save_txt
```

#### YOLOE (Text Prompt)
```bash
python3 baselines/yoloe/inference_text.py \
    -if /path/to/images \
    -tp "load" \
    --save_txt
```

## 📊 Evaluation

To evaluate the performance of the detection models, we use the **Mean Average Precision (mAP)** metric.

We utilize the [Cartucho/mAP](https://github.com/Cartucho/mAP) repository for computing these metrics.

**Steps:**
1.  Generate detection results using any of the methods above (ensure `--save_txt` is used).
2.  Clone the mAP repository:
    ```bash
    git clone https://github.com/Cartucho/mAP.git
    ```
3.  Copy your ground truth files into `mAP/input/ground-truth/`.
4.  Copy your generated detection files into `mAP/input/detection-results/`.
5.  Run the evaluation script:
    ```bash
    python mAP/main.py
    ```

## 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wacv2026submission,
  title={Few-Shot and Zero-Shot Object Detection for Industrial Lateral Loads},
  author={Losada del Olmo, Juan Jesús and Pardo Ballesteros, Emilio and López-de-Teruel, Pedro E. and Ruiz, Alberto},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
