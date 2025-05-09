# Robust Watermark Detection using Tamper-Aware Training for AI-Generated Image Integrity
This project explores a lightweight deep learning system for detecting invisible watermarks in AI-generated images—even when those images have been tampered with or attacked. It combines **Stable Diffusion**, **Meta's Stable Signature**, and a tamper-aware **MobileNetV3-Small** classifier to assess watermark presence under multiple distortion scenarios.

---

## Project Overview

- **Goal**: Build a binary classifier to detect watermark presence in AI-generated images, even after quality degradation or adversarial attacks.
- **Why It Matters**: Existing watermarking systems often break when images are compressed, blurred, or attacked. This work explores a **novel detection strategy** that trains models on **tampered watermarked images**, improving robustness.
- **Key Insight**: Rather than relying on traditional watermark decoding, this method detects the presence of watermark patterns learned during training.

---

## Repository Structure
.
├── build_dataset.py # Generate and label training/validation/test datasets
├── build_unseen_dataset.py # Create compound-tampered evaluation set
├── generate_images.py # Generate watermarked and non-watermarked images via Stable Diffusion
├── generate_prompts.py # Text prompts for image generation
├── apply_tampering.py # JPEG, noise, blur, crop tampering functions
├── attack_mobilenet.py # Adversarial attack generation (FGSM, PGD)
├── mobilenet_model.py # MobileNetV3-Small classifier architecture
├── train_baseline.py # Train on untampered data
├── train_experimental.py # Train on tampered + untampered data
├── train_finetuned.py # Fine-tune baseline on tampered data
├── evaluate_model.py # Evaluate models and generate confusion matrices, metrics
├── visualize_feature_maps.py # Visualize intermediate Conv layer activations
└── run_detection_pipeline.py # Full detection pipeline orchestration

---

## Datasets

Three main datasets are used:

1. **Baseline Dataset** – 12,000 untampered images (50% watermarked)
2. **Experimental Dataset** – 15,000 images, including 5 tampering types (JPEG, noise, blur, crop, adversarial)
3. **Unseen Dataset** – 600 images with randomly combined distortions to test generalization

Watermarked images are generated using Stable Signature (decoder finetuned on a binary signature), embedded into Stable Diffusion outputs.

---

## Model: MobileNetV3-Small

- **Architecture**: Pretrained on ImageNet, modified for binary classification
- **Training Variants**:
  - **Baseline**: Clean data only
  - **Experimental**: Clean + tampered images
  - **Finetuned**: Baseline model fine-tuned on tampered data (classifier unfrozen)

**Training Parameters**:
- Epochs: 40 (with early stopping)
- Optimizer: Adam
- Loss: CrossEntropy
- Batch Size: 32
- Input Size: 224x224

---

## Evaluation

Models are evaluated on:

- Clean test data
- Individually tampered samples (JPEG, noise, etc.)
- Compound tampered samples (unseen combinations)

**Metrics:**
- Accuracy, Precision, Recall, F1 Score
- False Negative Rate (FNR), False Positive Rate (FPR)
- Confusion matrices
- Tamper-type breakdowns
