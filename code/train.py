"""
Training Script for Google Colab / Kaggle
==========================================
MobileNetV2 full fine-tune for wafer defect classification.


The script will:
  - Fine-tune MobileNetV2 (ImageNet) with ALL layers unfrozen
  - Phase 1: Train classification head (base frozen)
  - Phase 2: Unfreeze all conv layers, fine-tune with small LR
  - Optionally apply QAT (Quantization-Aware Training)
  - Export TFLite FP32/FP16/INT8 models
  - Save confusion matrix, metrics, and all outputs to outputs/
"""

import os
import json
import random
import sys
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════
CLASSES = ["scratch", "block_etch", "particle",
           "coating_bad", "piq_particle", "sez_burnt", "clean", "other"]
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 1)

# Paths — adjust if needed
DATA_DIR = Path("/kaggle/input/mb-dt-v6-224/data/final")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
EXPORT_DIR = OUTPUT_DIR / "exports"
RESULTS_DIR = OUTPUT_DIR / "results"

# Training — two-phase approach
BATCH_SIZE = 32
WARMUP_EPOCHS = 20       # Phase 1: head only (base frozen)
FINETUNE_EPOCHS = 100    # Phase 2: all layers unfrozen
EPOCHS = WARMUP_EPOCHS + FINETUNE_EPOCHS
LEARNING_RATE = 1e-3     # Phase 1 LR (head training)
FINETUNE_LR = 5e-5       # Phase 2 LR (adapt pretrained features more aggressively)
DROPOUT_RATE = 0.3
MIXUP_ALPHA = 0.3          # Mixup blending strength (0 = off)
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MOBILENET_ALPHA = 0.35     # Width multiplier: 0.35 → ~500KB INT8 (fits NXP i.MX RT)
SEED = 42

# QAT
QAT_EPOCHS = 15
QAT_LEARNING_RATE = 5e-5

ALGORITHM_NAME = f"MobileNetV2-alpha{MOBILENET_ALPHA} (full fine-tune, all layers)"

# ══════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for d in [MODEL_DIR, EXPORT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE


# ══════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════
def _count_images(split_dir: Path) -> int:
    return sum(
        1
        for p in split_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _make_split_ds(split: str, shuffle: bool):
    split_dir = DATA_DIR / split
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        color_mode="grayscale",
        batch_size=None,  # load unbatched for per-image augmentation
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    def _to_float01(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    ds = ds.map(_to_float01, num_parallel_calls=AUTOTUNE)
    return ds


@tf.function
def _augment(image, label):
    """Heavy per-image augmentation for training — runs in tf.data pipeline."""
    H, W = IMAGE_SIZE

    # Random flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random 90-degree rotations (wafer defects have no preferred orientation)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    # Random brightness & contrast
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.75, 1.25)

    # Random zoom: scale up 0-25%, then crop back to HxW
    scale = tf.random.uniform([], 1.0, 1.25)
    new_h = tf.cast(tf.cast(H, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)
    image = tf.image.resize(image, [new_h, new_w])
    image = tf.image.random_crop(image, [H, W, 1])

    # Random Erasing (cutout): mask a random patch with mean pixel value
    # Forces the model to use global context, not just local features
    if tf.random.uniform([]) < 0.5:
        eh = tf.random.uniform([], H // 8, H // 3, dtype=tf.int32)
        ew = tf.random.uniform([], W // 8, W // 3, dtype=tf.int32)
        ey = tf.random.uniform([], 0, H - eh, dtype=tf.int32)
        ex = tf.random.uniform([], 0, W - ew, dtype=tf.int32)
        mean_val = tf.reduce_mean(image)
        mask = tf.ones([eh, ew, 1]) * mean_val
        paddings = [[ey, H - ey - eh], [ex, W - ex - ew], [0, 0]]
        mask_full = tf.pad(mask, paddings)
        # Create binary mask: 0 in erased region, 1 elsewhere
        binary = tf.pad(tf.zeros([eh, ew, 1]), paddings, constant_values=1.0)
        image = image * binary + mask_full * (1.0 - binary)

    # Gaussian noise (simulates sensor noise)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.03)
    image = image + noise

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

print("\n--- TRAIN ---")
train_ds_raw = _make_split_ds("train", shuffle=True)
train_samples = _count_images(DATA_DIR / "train")
print(f"  Total: {train_samples}")

print("\n--- VAL ---")
val_ds_raw = _make_split_ds("val", shuffle=False)
val_samples = _count_images(DATA_DIR / "val")
print(f"  Total: {val_samples}")

print("\n--- TEST ---")
test_ds_raw = _make_split_ds("test", shuffle=False)
test_samples = _count_images(DATA_DIR / "test")
print(f"  Total: {test_samples}")

# Mixup: blends pairs of images and labels within a batch.
# Creates soft decision boundaries — reduces overconfident misclassifications
# on hard pairs like piq_particle/particle and block_etch/coating_bad.
def _mixup_batch(images, labels):
    """Apply Mixup augmentation to a batch of images and labels."""
    if MIXUP_ALPHA <= 0:
        return images, labels
    batch_size = tf.shape(images)[0]
    # Sample lambda from Beta(alpha, alpha)
    lam = tf.random.uniform([batch_size, 1, 1, 1], 0.0, MIXUP_ALPHA)
    # Shuffle indices for pairing
    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)
    # Blend images
    mixed_images = images * (1.0 - lam) + images_shuffled * lam
    # Blend one-hot labels
    labels_oh = tf.one_hot(tf.cast(labels, tf.int32), NUM_CLASSES)
    labels_shuffled_oh = tf.one_hot(tf.cast(labels_shuffled, tf.int32), NUM_CLASSES)
    lam_labels = tf.reshape(lam, [batch_size, 1])
    mixed_labels = labels_oh * (1.0 - lam_labels) + labels_shuffled_oh * lam_labels
    return mixed_images, mixed_labels


# Convert sparse labels to one-hot (needed for Mixup compatibility)
def _to_onehot(x, y):
    return x, tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)


# Build pipelines: augmentation + mixup on training data
train_ds = (
    train_ds_raw
    .map(_augment, num_parallel_calls=AUTOTUNE)
    .shuffle(2048, seed=SEED)
    .batch(BATCH_SIZE)
    .map(_mixup_batch, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_ds_raw
    .map(_to_onehot, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
)
test_ds = (
    test_ds_raw
    .map(_to_onehot, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
)

print(f"\n  Pipeline: augment -> shuffle -> batch({BATCH_SIZE}) -> mixup -> prefetch")

# Compute class weights (handles imbalanced 'other' and 'sez_burnt')
_class_counts = {}
for i, cls_name in enumerate(CLASSES):
    cls_dir = DATA_DIR / "train" / cls_name
    _class_counts[i] = sum(
        1 for p in cls_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
_total = sum(_class_counts.values())
class_weight = {
    i: _total / (NUM_CLASSES * count) for i, count in _class_counts.items()
}
print("\n  Class weights:")
for i, cls_name in enumerate(CLASSES):
    print(f"    {cls_name}: {class_weight[i]:.3f} ({_class_counts[i]} train images)")


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════
def _collect_y_true(ds: tf.data.Dataset) -> np.ndarray:
    y_list = []
    for _, y in ds:
        yb = y.numpy()
        if yb.ndim == 2:  # one-hot → argmax to get integer labels
            yb = np.argmax(yb, axis=-1)
        y_list.append(yb)
    return np.concatenate(y_list, axis=0)


def _collect_y_pred(model_: tf.keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    probs = model_.predict(ds, verbose=0)
    return np.argmax(probs, axis=1).astype(np.int32)


def _save_confusion_matrix(cm: np.ndarray, out_png: Path, out_csv: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    header = ",".join(["true\\pred"] + CLASSES)
    with open(out_csv, "w") as f:
        f.write(header + "\n")
        for i, row in enumerate(cm):
            f.write(",".join([CLASSES[i]] + [str(int(v)) for v in row]) + "\n")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _tflite_predict_class(interpreter: tf.lite.Interpreter, x01: np.ndarray) -> int:
    """Run 1-sample inference. x01 is float32 in [0,1], shape (128,128,1)."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = x01[None, ...]  # add batch dim
    in_dtype = input_details["dtype"]

    if in_dtype == np.uint8 or in_dtype == np.int8:
        scale, zero_point = input_details.get("quantization", (1.0, 0))
        if scale == 0:
            scale = 1.0
        x_q = np.clip(np.round(x / scale + zero_point), 0, 255).astype(in_dtype)
        interpreter.set_tensor(input_details["index"], x_q)
    else:
        interpreter.set_tensor(input_details["index"], x.astype(in_dtype))

    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])[0]
    return int(np.argmax(out))


def _evaluate_tflite_model(tflite_path: Path, ds: tf.data.Dataset) -> dict:
    """Compute top-1 accuracy for a TFLite model over a dataset."""
    if not tflite_path.exists():
        return {"path": str(tflite_path), "exists": False}

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    correct = 0
    total = 0
    for x_batch, y_batch in ds:
        xb = x_batch.numpy()
        yb = y_batch.numpy()
        if yb.ndim == 2:  # one-hot → integer
            yb = np.argmax(yb, axis=-1)
        yb = yb.astype(np.int32)
        for i in range(xb.shape[0]):
            pred = _tflite_predict_class(interpreter, xb[i])
            correct += int(pred == int(yb[i]))
            total += 1

    acc = float(correct / max(1, total))
    in_dtype = interpreter.get_input_details()[0]["dtype"].__name__
    out_dtype = interpreter.get_output_details()[0]["dtype"].__name__
    return {
        "path": str(tflite_path),
        "exists": True,
        "accuracy": acc,
        "samples": int(total),
        "input_dtype": in_dtype,
        "output_dtype": out_dtype,
    }


# ══════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════
def _make_loss():
    """CategoricalCrossentropy with label smoothing. All labels are one-hot."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if LABEL_SMOOTHING and LABEL_SMOOTHING > 0:
            y_true = y_true * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / NUM_CLASSES)
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        )
    return loss_fn


# ══════════════════════════════════════════════
# MODEL: MobileNetV2 full fine-tune
# ══════════════════════════════════════════════
def build_mobilenetv2():
    """
    MobileNetV2 with ImageNet weights, adapted for grayscale wafer defect images.

    Strategy:
      - Grayscale [0,1] → RGB [-1,1] conversion inside the model
      - MobileNetV2 backbone (all layers will be unfrozen in Phase 2)
      - Classification head: GAP → BN → Dropout → Dense(8)
      - alpha=0.35: ~0.4M params → INT8 ~500KB (fits NXP i.MX RT)

    Two-phase training:
      Phase 1: Freeze base, train head only → learn class boundaries
      Phase 2: Unfreeze ALL conv layers (BN frozen) → adapt all features
    """
    input_layer = layers.Input(shape=INPUT_SHAPE, name="grayscale_input")

    # Grayscale [0,1] → RGB: repeat channel 3 times
    x = layers.Concatenate(name="gray_to_rgb")(
        [input_layer, input_layer, input_layer]
    )
    # Scale to MobileNetV2 expected range: [0,1] → [-1,1]
    x = layers.Lambda(lambda t: t * 2.0 - 1.0, name="rescale_neg1_1")(x)

    # MobileNetV2 backbone (alpha controls width multiplier → model size)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        alpha=MOBILENET_ALPHA,
        include_top=False,
        weights="imagenet",
    )
    x = base_model(x)

    # Classification head — compact for alpha=0.35 backbone
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(DROPOUT_RATE, name="head_drop")(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = Model(inputs=input_layer, outputs=output, name="mobilenetv2_finetune")
    return model, base_model


print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)
model, base_model = build_mobilenetv2()
model.summary()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"mbv2_finetune_{timestamp}"


# ══════════════════════════════════════════════
# PHASE 1: Train head only (base frozen)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"PHASE 1: Train head ({WARMUP_EPOCHS} epochs, base frozen)")
print("=" * 60)

base_model.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=_make_loss(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
)

cb_phase1 = [
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / f"{run_name}_best.keras"),
        monitor="val_accuracy", save_best_only=True, mode="max", verbose=1,
    ),
]

history1 = model.fit(
    train_ds,
    epochs=WARMUP_EPOCHS,
    validation_data=val_ds,
    callbacks=cb_phase1,
    verbose=1,
)


# ══════════════════════════════════════════════
# PHASE 2: Unfreeze ALL layers, fine-tune
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"PHASE 2: Full fine-tune ({FINETUNE_EPOCHS} epochs, all layers)")
print("=" * 60)

# Unfreeze all conv/dense layers; keep BatchNorm frozen (preserves pretrained stats)
base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
total_count = len(base_model.layers)
print(f"  Base model: {trainable_count}/{total_count} layers trainable (BN frozen)")

# Cosine decay from FINETUNE_LR to near-zero
steps_per_epoch = max(1, train_samples // BATCH_SIZE)
ft_total_steps = steps_per_epoch * FINETUNE_EPOCHS
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=FINETUNE_LR,
    decay_steps=ft_total_steps,
    alpha=1e-7,
)

# AdamW with weight decay
try:
    opt = optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
except Exception:
    print("[INFO] AdamW not available, using Adam")
    opt = optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=opt,
    loss=_make_loss(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
)

cb_phase2 = [
    callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / f"{run_name}_best.keras"),
        monitor="val_accuracy", save_best_only=True, mode="max", verbose=1,
    ),
    callbacks.EarlyStopping(
        monitor="val_accuracy", patience=20, restore_best_weights=True, verbose=1,
    ),
]

history2 = model.fit(
    train_ds,
    epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS,
    initial_epoch=WARMUP_EPOCHS,
    validation_data=val_ds,
    callbacks=cb_phase2,
    verbose=1,
)

# Save final FP32 model
final_path = MODEL_DIR / f"{run_name}_final.keras"
model.save(str(final_path))
print(f"\n[OK] FP32 model saved: {final_path}")

# Test evaluation
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"[RESULT] FP32 Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")


# ══════════════════════════════════════════════
# QAT (OPTIONAL)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Quantization-Aware Training (QAT)")
print("=" * 60)

try:
    import tensorflow_model_optimization as tfmot

    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=optimizers.Adam(learning_rate=QAT_LEARNING_RATE),
        loss=_make_loss(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    qat_cb = [
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "qat_best.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max", verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
    ]

    qat_history = qat_model.fit(
        train_ds,
        epochs=QAT_EPOCHS,
        validation_data=val_ds,
        callbacks=qat_cb,
        verbose=1,
    )

    qat_loss, qat_acc = qat_model.evaluate(test_ds, verbose=0)
    print(f"\n[RESULT] QAT Test Accuracy: {qat_acc:.4f} (FP32 baseline: {test_acc:.4f})")
    print(f"[RESULT] Delta: {qat_acc - test_acc:+.4f}")

    qat_model.save(str(MODEL_DIR / "qat_final.keras"))

except ImportError:
    print("[WARN] tensorflow-model-optimization not installed. Skipping QAT.")
    print("       Install: pip install tensorflow-model-optimization")
    qat_model = None
    qat_acc = None


# ══════════════════════════════════════════════
# EXPORT TFLITE
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("TFLite Export")
print("=" * 60)


def representative_dataset():
    """For INT8 calibration."""
    rep_ds = train_ds.unbatch().map(lambda x, y: x).batch(1).take(200)
    for x in rep_ds:
        yield [x]


# FP32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_fp32 = converter.convert()
fp32_path = EXPORT_DIR / "model_fp32.tflite"
with open(fp32_path, "wb") as f:
    f.write(tflite_fp32)
print(f"  [FP32] {len(tflite_fp32)/1024:.1f} KB -> {fp32_path}")

# FP16
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
fp16_path = EXPORT_DIR / "model_fp16.tflite"
with open(fp16_path, "wb") as f:
    f.write(tflite_fp16)
print(f"  [FP16] {len(tflite_fp16)/1024:.1f} KB -> {fp16_path}")

# INT8 PTQ
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_int8_ptq = converter.convert()
int8_ptq_path = EXPORT_DIR / "model_int8_ptq.tflite"
with open(int8_ptq_path, "wb") as f:
    f.write(tflite_int8_ptq)
print(f"  [INT8-PTQ] {len(tflite_int8_ptq)/1024:.1f} KB -> {int8_ptq_path}")

# INT8 QAT (if available)
if qat_model is not None:
    converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_int8_qat = converter.convert()
    int8_qat_path = EXPORT_DIR / "model_int8_qat.tflite"
    with open(int8_qat_path, "wb") as f:
        f.write(tflite_int8_qat)
    print(f"  [INT8-QAT] {len(tflite_int8_qat)/1024:.1f} KB -> {int8_qat_path}")


# TFLite inference evaluation
print("\n" + "=" * 60)
print("TFLite Inference Evaluation (Test Set)")
print("=" * 60)

tflite_metrics = {
    "fp32": _evaluate_tflite_model(fp32_path, test_ds),
    "fp16": _evaluate_tflite_model(fp16_path, test_ds),
    "int8_ptq": _evaluate_tflite_model(int8_ptq_path, test_ds),
}
if qat_model is not None:
    tflite_metrics["int8_qat"] = _evaluate_tflite_model(int8_qat_path, test_ds)

print(f"  [TFLite-FP32]  acc={tflite_metrics['fp32'].get('accuracy', None)}")
print(f"  [TFLite-FP16]  acc={tflite_metrics['fp16'].get('accuracy', None)}")
print(f"  [TFLite-INT8]  acc={tflite_metrics['int8_ptq'].get('accuracy', None)}")
if "int8_qat" in tflite_metrics:
    print(f"  [TFLite-QAT]   acc={tflite_metrics['int8_qat'].get('accuracy', None)}")


# ══════════════════════════════════════════════
# METRICS + CONFUSION MATRIX (TEST SET)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test-set metrics + confusion matrix")
print("=" * 60)

y_true = _collect_y_true(test_ds)
y_pred = _collect_y_pred(model, test_ds)

acc = float(accuracy_score(y_true, y_pred))
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
_save_confusion_matrix(
    cm,
    RESULTS_DIR / "confusion_matrix_test.png",
    RESULTS_DIR / "confusion_matrix_test.csv",
)

report_txt = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    digits=4,
    zero_division=0,
)
with open(RESULTS_DIR / "classification_report_test.txt", "w") as f:
    f.write(report_txt)

print(f"[RESULT] Test Accuracy: {acc:.4f}")
print(f"[RESULT] Macro Precision/Recall/F1: {prec_macro:.4f} / {rec_macro:.4f} / {f1_macro:.4f}")
print(f"[RESULT] Weighted Precision/Recall/F1: {prec_weighted:.4f} / {rec_weighted:.4f} / {f1_weighted:.4f}")


# ══════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════
results = {
    "run_name": run_name,
    "classes": CLASSES,
    "num_classes": NUM_CLASSES,
    "image_size": list(IMAGE_SIZE),
    "train_samples": int(train_samples),
    "val_samples": int(val_samples),
    "test_samples": int(test_samples),
    "algorithm": {
        "name": ALGORITHM_NAME,
        "architecture": f"MobileNetV2-alpha{MOBILENET_ALPHA} + GAP + BN + Dropout + Dense({NUM_CLASSES})",
        "transfer_learning": True,
        "input": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} grayscale, float32 [0,1]",
        "num_classes": int(NUM_CLASSES),
        "total_params": int(model.count_params()),
    },
    "platform": {
        "python": sys.version,
        "tensorflow": tf.__version__,
        "os": platform.platform(),
        "device": {
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
            "tpus": os.environ.get("COLAB_TPU_ADDR") is not None,
        },
        "environment": {
            "colab": os.environ.get("COLAB_RELEASE_TAG") is not None,
            "kaggle": os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None,
        },
        "inference": {
            "framework": "TensorFlow Lite",
            "device": "CPU",
        },
    },
    "test_metrics": {
        "accuracy": acc,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
    },
    "fp32_test_accuracy": float(test_acc),
    "fp32_test_loss": float(test_loss),
    "qat_test_accuracy": float(qat_acc) if qat_acc is not None else None,
    "model_sizes": {
        "fp32_kb": len(tflite_fp32) / 1024,
        "fp16_kb": len(tflite_fp16) / 1024,
        "int8_ptq_kb": len(tflite_int8_ptq) / 1024,
    },
    "tflite_test_metrics": tflite_metrics,
}
with open(RESULTS_DIR / "training_results.json", "w") as f:
    json.dump(results, f, indent=2)

env_name = "Colab" if results["platform"]["environment"]["colab"] else (
    "Kaggle" if results["platform"]["environment"]["kaggle"] else "Local"
)

with open(RESULTS_DIR / "model_results_summary.md", "w") as f:
    f.write("# Model Results (Test Set)\n\n")
    f.write(f"- **Algorithm**: {ALGORITHM_NAME}\n")
    f.write(f"- **Parameters**: {model.count_params():,}\n")
    f.write(f"- **Classes**: {NUM_CLASSES}\n")
    f.write(f"- **Training Platform**: {env_name}\n")
    f.write(f"- **Inference Platform**: TensorFlow Lite (CPU)\n\n")
    f.write("## Metrics\n\n")
    f.write(f"| Metric | Value |\n|---|---|\n")
    f.write(f"| Accuracy | {acc:.4f} |\n")
    f.write(f"| Precision (macro) | {prec_macro:.4f} |\n")
    f.write(f"| Recall (macro) | {rec_macro:.4f} |\n")
    f.write(f"| F1 (macro) | {f1_macro:.4f} |\n")
    f.write(f"| Precision (weighted) | {prec_weighted:.4f} |\n")
    f.write(f"| Recall (weighted) | {rec_weighted:.4f} |\n")
    f.write(f"| F1 (weighted) | {f1_weighted:.4f} |\n\n")
    f.write("## Artifacts\n\n")
    f.write("- Confusion Matrix: `outputs/results/confusion_matrix_test.png`\n")
    f.write("- Confusion Matrix CSV: `outputs/results/confusion_matrix_test.csv`\n")
    f.write("- Classification Report: `outputs/results/classification_report_test.txt`\n\n")
    f.write("## Model Sizes\n\n")
    f.write(f"| Format | Size |\n|---|---|\n")
    f.write(f"| FP32 | {results['model_sizes']['fp32_kb']:.1f} KB |\n")
    f.write(f"| FP16 | {results['model_sizes']['fp16_kb']:.1f} KB |\n")
    f.write(f"| INT8 (PTQ) | {results['model_sizes']['int8_ptq_kb']:.1f} KB |\n\n")
    f.write("## TFLite Inference (Test)\n\n")
    f.write(f"| Variant | Accuracy | Input | Output |\n|---|---|---|---|\n")
    for k, v in tflite_metrics.items():
        if not v.get("exists"):
            continue
        f.write(f"| {k} | {v.get('accuracy', 0):.4f} | {v.get('input_dtype')} | {v.get('output_dtype')} |\n")

# Save training config for reproduction
config = {
    "algorithm": ALGORITHM_NAME,
    "classes": CLASSES,
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "dropout_rate": DROPOUT_RATE,
    "weight_decay": WEIGHT_DECAY,
    "label_smoothing": LABEL_SMOOTHING,
    "mobilenet_alpha": MOBILENET_ALPHA,
    "mixup_alpha": MIXUP_ALPHA,
    "warmup_epochs": WARMUP_EPOCHS,
    "finetune_epochs": FINETUNE_EPOCHS,
    "finetune_lr": FINETUNE_LR,
    "lr_schedule": "CosineDecay (Phase 2)",
    "optimizer": "Adam (Phase 1) / AdamW (Phase 2)",
    "augmentation": "flip, rot90, brightness, contrast, zoom, random_erasing, gaussian_noise, mixup",
    "qat_epochs": QAT_EPOCHS,
    "qat_learning_rate": QAT_LEARNING_RATE,
    "seed": SEED,
}
with open(RESULTS_DIR / "training_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'=' * 60}")
print("TRAINING COMPLETE!")
print(f"{'=' * 60}")
print(f"  Algorithm:      {ALGORITHM_NAME}")
print(f"  FP32 Accuracy:  {test_acc:.4f}")
if qat_acc is not None:
    print(f"  QAT Accuracy:   {qat_acc:.4f}")
print(f"  Models:         {MODEL_DIR}")
print(f"  TFLite exports: {EXPORT_DIR}")
print(f"  Results:        {RESULTS_DIR}")
print(f"\nDownload outputs/ folder and place back in project!")
