# Imports
import os
import torch
from tqdm import tqdm
from PIL import Image
import csv
import random
import numpy as np
import cv2

from apply_tampering import jpeg_compression, gaussian_noise, blur_image, crop_image
from attack_mobilenet import adversarial_attack
from mobilenet_model import WatermarkDetectionMobileNetV3

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load watermark detection model
model = WatermarkDetectionMobileNetV3()
model.load_state_dict(torch.load("models/mobilenet_watermark_detector.pth", map_location=device))
model = model.to(device).eval()

# Define all tampering methods and strengths
ALL_TAMPER_METHODS = [
    (jpeg_compression, [90, 70, 50, 30, 10]),
    (gaussian_noise, [5, 10, 15, 20, 25]),
    (blur_image, [3, 5, 7, 9, 11]),
    (crop_image, [0.05, 0.1, 0.15, 0.2, 0.25]),
    ("fgsm", [0.01, 0.03, 0.05, 0.07, 0.10]),
    ("pgd", [0.01, 0.015, 0.02, 0.025, 0.03])
]

# Define function to apply a single tampering method to an image
def apply_tamper(img, method, param):
    if method == "fgsm" or method == "pgd":
        return adversarial_attack(img, model, attack_type=method, epsilon=param, device=device)
    else:
        return method(img, param)

# Define function to apply a random combination of tampering methods to each image
def apply_multiple_perturbations(input_dir, output_dir, max_samples=300, min_combo=2, max_combo=6):
    os.makedirs(output_dir, exist_ok=True)
    images = sorted(os.listdir(input_dir))[:max_samples]
    for i, fname in enumerate(images):
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        
        # Choose a random number of tamper methods to apply
        n_comb = random.randint(min_combo, max_combo)
        
        # Choose random types of tamper methods to apply
        chosen_methods = random.sample(ALL_TAMPER_METHODS, k=n_comb)

        # Initilize list to keep track of which methods and strengths were applied
        combo_str = []
        for method, options in chosen_methods:
            param = random.choice(options)
            method_name = method if isinstance(method, str) else method.__name__
            combo_str.append(f"{method_name}_{param}")
            img = apply_tamper(img, method, param)

        # Save output
        out_fname = f"multi_{'__'.join(combo_str)}_{i:04d}.png"
        img.save(os.path.join(output_dir, out_fname))

# Define function to tag all generated combined tampered images into a CSV for tracking and labeling
def tag_dataset_to_csv(split_dir, split_name, csv_path="unseen_dataset_labels.csv"):
    rows = []
    for wm_status in ["watermarked", "non_watermarked"]:
        label_val = 0 if wm_status == "watermarked" else 1
        base_dir = os.path.join(split_dir, wm_status, "tampered", "combined")
        if not os.path.exists(base_dir):
            continue
        for fname in sorted(os.listdir(base_dir)):
            if fname.lower().endswith(".png"):
                rows.append({
                    "filename": os.path.join(split_name, wm_status, "tampered", "combined", fname),
                    "label": label_val,
                    "tampered": True,
                    "tamper_type": "combined",
                    "tamper_strength": "multi",
                    "split": split_name
                })
    
    # Write metadata to CSV 
    fieldnames = ["filename", "label", "tampered", "tamper_type", "tamper_strength", "split"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # Print how many images had metadata logged and where they were saved
    print(f"Logged {len(rows)} images from split '{split_name}' to {csv_path}")

# Execute only if the script is run directly
if __name__ == "__main__":
    # Define output structure for new tampered test dataset
    BASE_DIR = "unseen_dataset"
    SPLIT_NAME = "unseen_test_combined"
    os.makedirs(BASE_DIR, exist_ok=True)

    # Get untampered images from experimental_test split
    wm_in = "custom_dataset/experimental_test/watermarked/untampered"
    nwm_in = "custom_dataset/experimental_test/non_watermarked/untampered"

    # Define output directories for saving combined tampered images
    wm_out = os.path.join(BASE_DIR, SPLIT_NAME, "watermarked", "tampered", "combined")
    nwm_out = os.path.join(BASE_DIR, SPLIT_NAME, "non_watermarked", "tampered", "combined")

    # Apply multiple tampering methods to both watermarked and non-watermarked images
    apply_multiple_perturbations(wm_in, wm_out, max_samples=300)
    apply_multiple_perturbations(nwm_in, nwm_out, max_samples=300)

    # Log all new tampered samples into the unseen dataset CSV
    tag_dataset_to_csv(os.path.join(BASE_DIR, SPLIT_NAME), SPLIT_NAME, csv_path="unseen_dataset_labels.csv")
