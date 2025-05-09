# Imports
import os
import torch
from PIL import Image
from generate_prompts import get_prompts
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torchvision.utils import save_image
from ldm.models.diffusion.ddim import DDIMSampler

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define function to load the Stable Diffusion model from a configuration and checkpoint
def load_model(config_path, checkpoint_path):
    # Load configuration file
    config = OmegaConf.load(config_path)
    
    # Create model using only the model config section
    model = instantiate_from_config(config.model)
    
    # Load model weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights into model
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    # Move to device and set model to evaluation mode
    return model.to(device).eval()

# Define function to override the model's decoder with a fine-tuned decoder checkpoint
def override_decoder(model, decoder_ckpt_path):
    decoder_ckpt = torch.load(decoder_ckpt_path, map_location=device)
    
    # Extract the decoder portion of the checkpoint
    decoder_state = decoder_ckpt["ldm_decoder"]
    
    # Replace decoder weights
    model.first_stage_model.load_state_dict(decoder_state, strict=False)
    
    return model

# Define function for generating images using DDIM sampling from prompts
def generate_images(model, prompts, output_dir, prefix, num_images):
    # Create sampler instance using the model
    sampler = DDIMSampler(model)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define sampling hyperparameters
    batch_size = 1 # Number of images per prompt
    ddim_steps = 50 # Number of diffusion steps
    scale = 7.5  # Classifier-free guidance scale
    shape = [4, 64, 64] # Latent space shape for sampling

    for i in range(num_images):
        # Cycle through prompts
        prompt = prompts[i % len(prompts)]
        
        # Define unconditional conditioning
        uc = model.get_learned_conditioning(batch_size * [""])
        
        # Define conditional conditioning with prompt
        c = model.get_learned_conditioning(batch_size * [prompt])

        # Run DDIM sampling
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=0.0, # No added noise
        )
        
        # Decode latent image back to pixel space
        x_samples = model.decode_first_stage(samples_ddim)
        
        # Normalize from [-1, 1] to [0, 1]
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
        
        # Save output image
        img = x_samples[0]
        save_image(img, os.path.join(output_dir, f"{prefix}_{i:05d}.png"))
        print(f"Saved {prefix}_{i:05d}.png")

# Define function to generate watermarked images using the Stable Signature decoder
def generate_watermarked_images_sd_ss(ldm_config, ldm_ckpt, decoder_ckpt, key_path, output_dir, num_images):
    # Load base model
    model = load_model(ldm_config, ldm_ckpt)
    
    # Override decoder with watermark-specific decoder
    model = override_decoder(model, decoder_ckpt)

    # Load watermark key for metadata
    with open(key_path, "r") as f:
        watermark_key = f.readline().strip()
    print(f"Using watermark key: {watermark_key}")

    # Load prompts
    prompts = get_prompts()
    
    # Generate watermarked images
    generate_images(model, prompts, output_dir, "watermarked", num_images)

# Define function to generate non-watermarked images using Stable Diffusion
def generate_non_watermarked_images_sd(ldm_config, ldm_ckpt, output_dir, num_images):
    # Load base model
    model = load_model(ldm_config, ldm_ckpt)
    
    # Load prompts
    prompts = get_prompts()
    
    # Generate non-watermarked images
    generate_images(model, prompts, output_dir, "nonwatermarked", num_images)
