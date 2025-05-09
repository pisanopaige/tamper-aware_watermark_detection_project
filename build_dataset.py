# Imports
import os 
import torch
from tqdm import tqdm
from PIL import Image
from apply_tampering import jpeg_compression, gaussian_noise, blur_image, crop_image
from attack_mobilenet import adversarial_attack
from generate_images import generate_watermarked_images_sd_ss, generate_non_watermarked_images_sd
import csv

# Define function to apply tampering and save results
def apply_and_save(method_func, param_list, input_dir, output_dir, per_param_count):
    # Get all image names in the folder
    images = sorted(os.listdir(input_dir))
    
    # Make output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize count for saved images
    count = 0
    
    # Go through each tampering strength and apply each strength to the number of images
    for i, param in enumerate(param_list):
        for j in range(per_param_count):
            idx = i * per_param_count + j
            if idx >= len(images):  # Skip if there are no more images
                continue
            
            # Get image path
            img_path = os.path.join(input_dir, images[idx])
            
            # Load image in RGB mode
            img = Image.open(img_path).convert("RGB")
            
            # Apply tampering
            tampered = method_func(img, param)
            
            # Save output
            tampered.save(os.path.join(output_dir, f"{method_func.__name__}_{param}_{j:04d}.png"))
            
            # Add to the image save count
            count += 1
    
    # Print how many images were saved and where they were saved
    print(f"Saved {count} tampered images to {output_dir}")

# Define function to apply adversarial attacks and save results
def apply_adversarial_attacks(model, input_dir, output_dir, fgsm_count, pgd_count, device):
    # Get all image names in the folder
    images = sorted(os.listdir(input_dir))
    
    # Make output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define strengths of adversarial attacks
    fgsm_eps = [0.01, 0.03, 0.05, 0.07, 0.10]
    pgd_eps = [0.01, 0.015, 0.02, 0.025, 0.03]
    
    # Get the total amount of input images
    total_images = len(images)

    # Define the number of images per FGSM attack strength
    fgsm_per_eps = max(1, fgsm_count // len(fgsm_eps))
    
    # Initialize count for saved FGSM images
    fgsm_total = 0
    
    # Go through each tampering strength and apply each strength to the number of images
    for i, eps in enumerate(fgsm_eps):
        for j in range(fgsm_per_eps):
            img_idx = (i * fgsm_per_eps + j) % total_images # Reuse images from the start if there are less images than needed
            
            # Get image path
            img_path = os.path.join(input_dir, images[img_idx])
            
            # Load image in RGB mode
            img = Image.open(img_path).convert("RGB")
            
            # Run FGSM attack
            adv = adversarial_attack(img, model, "fgsm", epsilon=eps, device=device)
            if adv:
                # Save output
                adv.save(os.path.join(output_dir, f"fgsm_{eps:.3f}_{i:02d}_{j:04d}.png"))
                
                # Add to the FGSM image save count
                fgsm_total += 1

    # Define the number of images per PGD attack strength
    pgd_per_eps = max(1, pgd_count // len(pgd_eps))
    
    # Initialize count for saved PGD images
    pgd_total = 0
    
    # Go through each tampering strength and apply each strength to the number of images
    for i, eps in enumerate(pgd_eps):
        for j in range(pgd_per_eps):
            img_idx = (i * pgd_per_eps + j) % total_images  # Reuse images from the start if there are less images than needed
            
            # Get image path
            img_path = os.path.join(input_dir, images[img_idx])
            
            # Load image in RGB mode
            img = Image.open(img_path).convert("RGB")
            
            # Run PGD attack
            adv = adversarial_attack(img, model, "pgd", epsilon=eps, device=device)
            if adv:
                # Save output
                adv.save(os.path.join(output_dir, f"pgd_{eps:.3f}_{i:02d}_{j:04d}.png"))
                
                # Add to the PGD image save count
                pgd_total += 1

    # Print how many images were saved and where they were saved
    print(f"Saved {fgsm_total} FGSM and {pgd_total} PGD adversarial images to {output_dir}")

# Define function to tag all images with labels and metadata, and log to CSV
def tag_dataset_to_csv(split_dir, split_name, csv_path="dataset_labels.csv"):
    # Initialize list to store all rows to write to CSV
    rows = []
    
    # Go through both classes and assign labels
    for wm_status in ["watermarked", "non_watermarked"]:
        label_val = 0 if wm_status == "watermarked" else 1
        
        # Go through untampered and tampered subdirectories
        for ttype in ["untampered", "tampered"]:
            base_dir = os.path.join(split_dir, wm_status, ttype)
            
            # Check each tampering type if tampered
            if ttype == "tampered":
                for tamper in ["jpeg", "noise", "blur", "crop", "adv"]:
                    tamper_dir = os.path.join(base_dir, tamper)
                    if not os.path.exists(tamper_dir):
                        continue # Skip directory if it does nto exist
                    
                    # Go through all images in directory
                    for fname in sorted(os.listdir(tamper_dir)):
                        if fname.lower().endswith(".png"):
                            # Extract strength from filename if possible
                            strength = "unknown"
                            parts = fname.split('_')
                            try:
                                if tamper in ["jpeg", "noise", "blur", "crop"]:
                                    strength = parts[2]
                                elif tamper == "adv":
                                    strength = parts[1]
                            except IndexError:
                                pass # Pass if formatting is incorrect
                            
                            # Append image metadata to rows
                            rows.append({
                                "filename": os.path.join(split_name, wm_status, ttype, tamper, fname),
                                "label": label_val,
                                "tampered": True,
                                "tamper_type": tamper,
                                "tamper_strength": strength,
                                "split": split_name
                            })
            else:
                # Handle untampered images
                if not os.path.exists(base_dir):
                    continue
                for fname in sorted(os.listdir(base_dir)):
                    if fname.lower().endswith(".png"):
                        rows.append({
                            "filename": os.path.join(split_name, wm_status, ttype, fname),
                            "label": label_val,
                            "tampered": False,
                            "tamper_type": "none",
                            "tamper_strength": "none",
                            "split": split_name
                        })
    
    # Write all metadata to CSV
    fieldnames = ["filename", "label", "tampered", "tamper_type", "tamper_strength", "split"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    
    # Print how many images had metadata logged and where they were saved
    print(f"Logged {len(rows)} images from split '{split_name}' to {csv_path}")

# Define function to generate a complete dataset split
def generate_split(name, model, base_dir, config, device):
    # Print status update
    print(f"\nGenerating {name} split...")
    
    # Define path to the directory where this split will be saved
    split_dir = os.path.join(base_dir, name)
    
    # Define base output paths for both watermark categories
    wm_base = os.path.join(split_dir, "watermarked")
    nwm_base = os.path.join(split_dir, "non_watermarked")
    
    # Generate untampered watermarked and non-watermarked images using SD/SS pipeline
    generate_watermarked_images_sd_ss(
        config["ldm_config"], config["ldm_ckpt"], config["decoder_ckpt"],
        config["key_path"], os.path.join(wm_base, "untampered"), config["watermarked_count"]
    )
    generate_non_watermarked_images_sd(
        config["ldm_config"], config["ldm_ckpt"], os.path.join(nwm_base, "untampered"), config["non_watermarked_count"]
    )

    # Apply tampering if specified
    if config["tamper_per_type"] > 0:
        for method_func, param_list, method_name in [
            (jpeg_compression, [90, 70, 50, 30, 10], "jpeg"),
            (gaussian_noise, [5, 10, 15, 20, 25], "noise"),
            (blur_image, [3, 5, 7, 9, 11], "blur"),
            (crop_image, [0.05, 0.1, 0.15, 0.2, 0.25], "crop")
        ]:
            # Apply each tampering method
            apply_and_save(method_func, param_list,
                           os.path.join(wm_base, "untampered"),
                           os.path.join(wm_base, "tampered", method_name),
                           config["per_param_count"])
            apply_and_save(method_func, param_list,
                           os.path.join(nwm_base, "untampered"),
                           os.path.join(nwm_base, "tampered", method_name),
                           config["per_param_count"])
            
        # Apply adversarial attacks
        apply_adversarial_attacks(model, os.path.join(wm_base, "untampered"),
                                  os.path.join(wm_base, "tampered", "adv"),
                                  config["adv_fgsm"], config["adv_pgd"], device)
        apply_adversarial_attacks(model, os.path.join(nwm_base, "untampered"),
                                  os.path.join(nwm_base, "tampered", "adv"),
                                  config["adv_fgsm"], config["adv_pgd"], device)
    
    # Tag this split into the global CSV metadata index
    tag_dataset_to_csv(split_dir, name)
    
    # Print status update
    print(f"{name} split complete.")

# Execute only if the script is run directly
if __name__ == "__main__":
    # Import the watermark detection model
    from mobilenet_model import WatermarkDetectionMobileNetV3

    # Define base directory where all dataset splits will be saved
    BASE_DIR = "custom_dataset"
    os.makedirs(BASE_DIR, exist_ok=True) # Create directory if it does not exist

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print what type of device is being used
    print(f"Using device: {device}")

    # Initialize and load pretrained watermark detection model
    model = WatermarkDetectionMobileNetV3()
    model.load_state_dict(torch.load("models/mobilenet_watermark_detector.pth", map_location=device))
    model = model.to(device).eval() # Move to device and set to evaluation mode

    # Define base configuration for image generation and watermark decoding
    config_base = {
        "ldm_config": "configs/stable-diffusion/v1-inference.yaml",
        "ldm_ckpt": "models/sd-v1-4.ckpt",
        "decoder_ckpt": "models/checkpoint_000.pth",
        "key_path": "models/keys.txt"
    }

    '''# Generate small-scale test split for debugging
    generate_split("test_mini", model, BASE_DIR, {
        **config_base,
        "watermarked_count": 1, # untampered
        "non_watermarked_count": 1, # untampered
        "tamper_per_type": 5,
        "per_param_count": 1,
        "adv_fgsm": 5,
        "adv_pgd": 5
    }, device)'''
    
    # Generate baseline training split
    generate_split("baseline_train", model, BASE_DIR, {
        **config_base,
        "watermarked_count": 4800, # untampered
        "non_watermarked_count": 4800, # untampered
        "tamper_per_type": 0,
        "per_param_count": 0,
        "adv_fgsm": 0,
        "adv_pgd": 0
    }, device)
    
    # Generate baseline validation split
    generate_split("baseline_val", model, BASE_DIR, {
        **config_base,
        "watermarked_count": 1200, # untampered
        "non_watermarked_count": 1200, # untampered
        "tamper_per_type": 0,
        "per_param_count": 0,
        "adv_fgsm": 0,
        "adv_pgd": 0
    }, device)
    
    # Generate experimental training split
    generate_split("experimental_train", model, BASE_DIR, {
        **config_base,
        "watermarked_count": 1600, # untampered
        "non_watermarked_count": 1600, # untampered
        "tamper_per_type": 5,
        "per_param_count": 128,
        "adv_fgsm": 320,
        "adv_pgd": 320
    }, device)
    
    # Generate experimental validation split
    generate_split("experimental_val", model, BASE_DIR, {
        **config_base,
        "watermarked_count": 400, # untampered
        "non_watermarked_count": 400, # untampered
        "tamper_per_type": 5,
        "per_param_count": 32,
        "adv_fgsm": 80,
        "adv_pgd": 80
    }, device)
    
    # Generate experimental test split
    generate_split("experimental_test", model, BASE_DIR, {
        **config_base,
        "watermarked_count": 500, # untampered
        "non_watermarked_count": 500, # untampered
        "tamper_per_type": 5,
        "per_param_count": 40,
        "adv_fgsm": 100,
        "adv_pgd": 100
    }, device)
