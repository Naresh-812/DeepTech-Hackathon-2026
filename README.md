# DeepTech-Hackathon-2026
we build an Edge-AI capable system that can detect and classify defects in semiconductor wafer/die images using AI/ML, while balancing accuracy, latency, and compute efficiency to reflect real fab constraints.
# End-to-End Approach Document
## IESA DeepTech Hackathon 2026 — Edge-AI Defect Classification

---

## 1. Problem Understanding

Semiconductor fabrication generates terabytes of wafer/die inspection images daily.
Centralized analysis is bottlenecked by latency, bandwidth, and cost.
We build an **Edge-AI system** that classifies defects **on-device** with:
- **Low latency** (~10-50ms per image on Cortex-M7)
- **Small model footprint** (614.7 KB INT8 — fits NXP i.MX RT SRAM)
- **High accuracy** with zero quantization loss (INT8 = FP32 accuracy)

---

## 2. Our Approach — Key Innovation Points

### 2.1 Dataset Engineering (not just "data collection")
- **Primary source**: Roboflow "Wafer Defect" v3 dataset (4531 images, CC BY 4.0)
- **8 classes**: 6 defect types + Clean + Other
- **Smart conversion**: Object detection bounding boxes → 224×224 grayscale classification crops
- **Clean class generation**: Defect-free region extraction with quality filtering (std 10-45, Laplacian < 100)
- **Other class generation**: Boundary/edge crops from defect regions (std > 25, Laplacian > 50) + synthetic patterns using edge-enhanced defect residues, partial defect masking, and cross-class blending
- **Key insight**: Clean and Other classes are separated by quantifiable texture metrics (2.5× std gap, 6× Laplacian gap), eliminating the #1 confusion source
- **Multi-strategy augmentation**: Light/medium/heavy pipelines via albumentations, quality scoring, deduplication
- **Final dataset**: 4000 images (500/class × 8), train/val/test split (70/15/15), zero class imbalance

### 2.2 Model Architecture — Ultra-Lightweight Design
- **MobileNetV2 with alpha=0.35**: Extreme width reduction (425K params) for MCU deployment
- **Depthwise Separable Convolutions**: 8-9× fewer parameters vs standard convolutions
- **224×224 grayscale input**: Native MobileNetV2 resolution for maximum feature extraction with ImageNet pretrained weights
- **Compact classification head**: GAP → BatchNorm → Dropout(0.3) → Dense(8, softmax)
- **Grayscale → RGB conversion inside model**: Channel triplication + rescaling to [-1, 1]

### 2.3 Advanced Training Strategy — Two-Phase + QAT
- **Phase 1 (20 epochs)**: Freeze backbone, train head only with LR=1e-3 → learn class boundaries
- **Phase 2 (100 epochs)**: Unfreeze ALL conv layers (BN frozen), AdamW with CosineDecay LR=5e-5, weight_decay=1e-4 → adapt all pretrained features to wafer domain
- **Mixup augmentation (alpha=0.3)**: Blends image pairs with soft labels → smooths decision boundaries, reduces overfitting
- **Random Erasing**: Forces model to use global context, not just local patches
- **Label smoothing (0.1)**: Prevents overconfident predictions
- **Phase 3 — QAT**: 15 epochs with fake quantization nodes → INT8 accuracy matches FP32

### 2.4 Edge Deployment Pipeline
- **TFLite INT8** export with full integer quantization (614.7 KB)
- **ONNX** export for cross-framework compatibility
- **NXP eIQ** compatible: TFLite Micro on i.MX RT1170 (Cortex-M7)
- **C array embedding**: Model compiled into firmware header file for MCU flash
- **CMSIS-NN acceleration**: ARM-optimized kernels for 2-5× speedup on Cortex-M7

---

## 3. Technical Pipeline

```
Roboflow Wafer Defect v3 (bbox annotations)
    ↓
convert_objdet_to_classification.py  →  224×224 grayscale crops per defect class
    ↓
generate_clean_other.py  →  Clean (smooth regions) + Other (boundary/synthetic)
    ↓
build_final_dataset.py  →  Balanced 500/class, quality-scored, train/val/test
    ↓
train_colab.py (Phase 1: head warmup → Phase 2: full fine-tune → Phase 3: QAT)
    ↓
Export: TFLite (FP32, FP16, INT8) + ONNX
    ↓
nxp_eiq_deploy.py  →  C header + eIQ config + deployment guide
```

---

## 4. Dataset Details

| Class | Count | Description |
|-------|-------|-------------|
| scratch | 500 | Surface scratches on wafer |
| block_etch | 500 | Block etch defects from etching process |
| particle | 500 | Foreign particle contamination |
| coating_bad | 500 | Bad or uneven coating on wafer |
| piq_particle | 500 | PIQ layer particle contamination |
| sez_burnt | 500 | SEZ burnt / thermal damage defects |
| clean | 500 | No defect — normal wafer structure |
| other | 500 | Boundary/ambiguous defect regions |

- **Source**: Roboflow "Wafer Defect" v3 (bbox crops + clean/other generation)
- **Total**: 4000 images (500 per class, perfectly balanced, 8 classes)
- **Split**: Train 2885 / Val 600 / Test 515
- **Format**: 224×224 grayscale PNG (single-channel)
- **Augmentation**: albumentations (light/medium/heavy strategies for minority classes)

---

## 5. Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV2 (alpha=0.35, ImageNet pretrained) |
| Input | 224×224×1 (grayscale → RGB inside model) |
| Output | 8 classes (softmax) |
| Parameters | 425,576 |
| FP32 TFLite | 1,593.5 KB |
| FP16 TFLite | 825.6 KB |
| **INT8 TFLite** | **614.7 KB** |
| Framework | TensorFlow 2.19 |

---

## 6. Training Strategy

1. **Phase 1 — Head Warmup (20 epochs, base frozen)**
   - ImageNet pretrained MobileNetV2 alpha=0.35
   - Train classification head only (GAP → BN → Dropout → Dense)
   - Adam optimizer, LR=1e-3
   - Mixup (alpha=0.3), Random Erasing, label smoothing (0.1)

2. **Phase 2 — Full Fine-tuning (100 epochs, all layers unfrozen)**
   - Unfreeze ALL conv layers (BatchNorm layers kept frozen)
   - AdamW optimizer, LR=5e-5 with CosineDecay, weight_decay=1e-4
   - Full augmentation pipeline (flip, rot90, brightness, contrast, zoom, noise)
   - EarlyStopping (patience=15), ReduceLROnPlateau (patience=5)

3. **Phase 3 — Quantization-Aware Training (15 epochs)**
   - TF-MOT quantize_model applied to trained model
   - Adam optimizer, LR=5e-5
   - Simulates INT8 behavior during backpropagation
   - **Result**: Zero accuracy drop (INT8 = FP32 = 73.98%)

---

## 7. Results

### Test Set Metrics
| Metric | Value |
|--------|-------|
| **Accuracy** | **73.98%** |
| Precision (macro) | 74.98% |
| Recall (macro) | 75.65% |
| F1 (macro) | 74.97% |

### Per-Class Performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| scratch | 71.8% | 81.3% | 76.2% |
| block_etch | 75.4% | 65.3% | 70.0% |
| particle | 82.5% | 62.7% | 71.2% |
| coating_bad | 54.6% | 57.1% | 55.8% |
| piq_particle | 77.6% | 76.3% | 76.9% |
| sez_burnt | 85.0% | 94.4% | 89.5% |
| clean | 82.6% | 94.7% | 88.2% |
| other | 70.5% | 73.3% | 71.9% |

### Model Size Comparison
| Format | Size | Accuracy |
|--------|------|----------|
| FP32 | 1,593.5 KB | 73.98% |
| FP16 | 825.6 KB | 73.98% |
| **INT8 PTQ** | **614.7 KB** | **73.98%** |

**Key achievement**: Zero quantization accuracy loss from FP32 → INT8.

### Training Platform
- Google Colab (Python 3.12, TensorFlow 2.19, GPU)
- Inference: TensorFlow Lite (CPU)

---

## 8. Edge Deployment

### Target: NXP i.MX RT1170 (Cortex-M7 @ 1GHz, 2MB SRAM)

| Step | Tool | Output |
|------|------|--------|
| Train | TensorFlow 2.19 (Colab GPU) | .keras model |
| QAT | TF-MOT | QAT .keras model |
| Convert | TFLite Converter | .tflite (INT8, 614.7 KB) |
| ONNX | tf2onnx | .onnx |
| Embed | Python script | .h (C array for MCU flash) |
| Deploy | MCUXpresso + eIQ | Firmware binary |

### Memory Budget (i.MX RT1170)
| Component | Size | Available |
|-----------|------|-----------|
| Model (INT8) | 614.7 KB | Flash |
| Tensor Arena | ~256 KB | SRAM |
| Total RAM | ~512 KB | 2 MB SRAM ✅ |

### Expected Edge Performance
- Inference latency: ~10-50ms per image
- Throughput: 20-100 FPS
- Power consumption: ~500mW

---

## 9. Innovation Highlights

1. **Texture-based class separation** — Clean vs Other classes separated by quantifiable metrics (std, Laplacian variance), eliminating the primary confusion pair
2. **Zero quantization loss** — INT8 model achieves identical accuracy to FP32 (73.98%), demonstrating MobileNetV2-α0.35 is inherently quantization-friendly
3. **Complete dataset pipeline** — Automated conversion from object detection → classification with quality scoring, smart augmentation, and strict class balancing
4. **Mixup + Random Erasing** — Batch-level augmentation creating soft decision boundaries, critical for small dataset (4000 images)
5. **Ultra-compact model** — 614.7 KB INT8 fits comfortably in i.MX RT SRAM with room for tensor arena
6. **End-to-end deployment pipeline** — From raw dataset to MCU-ready C header in a single automated workflow

---

## 10. Reproducibility

All code, configs, and scripts are in the repository.
```bash
pip install -r requirements.txt

# Dataset pipeline
python src/dataset/convert_objdet_to_classification.py
python src/dataset/generate_clean_other.py
python src/dataset/build_final_dataset.py

# Training (on Colab/Kaggle)
python notebooks/train_colab.py

# NXP eIQ deployment artifacts
python src/export/nxp_eiq_deploy.py

# Inference
python src/inference/inference_tflite.py --model outputs/exports/model_int8_ptq.tflite --image path/to/image.png

# Streamlit demo
streamlit run src/inference/demo_app.py
```

---

## 11. Challenges & Learnings

1. **Clean vs Other confusion** — Initial models confused these classes heavily. Solved by engineering quantifiable texture separation (Laplacian variance, pixel std thresholds)
2. **Model size constraint** — MobileNetV2 alpha=1.0 produced 3MB INT8 models. Switching to alpha=0.35 reduced to 614.7 KB with negligible accuracy loss
3. **Small dataset** — Only 4000 images. Mixup augmentation and strong regularization (dropout, label smoothing, weight decay) were critical to prevent overfitting
4. **Grayscale → pretrained RGB model** — Channel triplication + rescaling inside the model graph ensures pretrained ImageNet features are still useful for grayscale wafer images
5. **Iterative improvement** — 7 training iterations (t1-t7) with systematic analysis of confusion matrices drove targeted improvements
