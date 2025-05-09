# Imports
import numpy as np
import cv2
from PIL import Image
import io

# Define function to apply JPEG compression at a specific quality level
def jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    # Create an in-memory byte buffer
    buffer = io.BytesIO()
    
    # Save image to buffer with JPEG compression
    img.save(buffer, format="JPEG", quality=quality)
    
    # Reset buffer pointer to the beginning
    buffer.seek(0)
    
    # Read compressed image from buffer
    return Image.open(buffer).convert("RGB")


# Define function to apply gaussian noise at a specific standard deviation
def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    # Convert image to float32 NumPy array
    arr = np.array(img, dtype=np.float32)
    
    # Generate gaussian noise
    noise = np.random.normal(0, sigma, arr.shape)
    
    # Add noise to image
    arr_noisy = arr + noise
    
    # Clip to valid pixel range and convert to uint8
    arr_noisy = np.clip(arr_noisy, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    return Image.fromarray(arr_noisy)

# Define function to apply gaussian blur at a specific kernel size
def blur_image(img: Image.Image, k: int) -> Image.Image:
    
    # Make sure kernel size is odd for OpenCV
    if k % 2 == 0:
        k += 1
    
    # Convert to NumPy array
    arr = np.array(img)
    
    # Apply Gaussian blur and convert back to PIL Image
    return Image.fromarray(cv2.GaussianBlur(arr, (k, k), 0))

# Define function to apply cropping at a specific percentage from each size
def crop_image(img: Image.Image, percent: float) -> Image.Image:
    # Define width and height
    w, h = img.size
    
    # Compute pixels to remove
    dx, dy = int(w * percent), int(h * percent)
    if dx >= w // 2 or dy >= h // 2:
        raise ValueError("Crop percent too large, results in empty image")
    
    # Perform cropping
    return img.crop((dx, dy, w - dx, h - dy))
