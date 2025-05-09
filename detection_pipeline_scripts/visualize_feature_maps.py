# Imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from mobilenet_model import WatermarkDetectionMobileNetV3


# Define function to visualize feature maps
def visualize_feature_maps(model_path, test_dir, output_prefix="model"):
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image preproccesing steps
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize image to expected input size
        transforms.ToTensor() # Convert to tensor
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    # Stop if there are no images in the test set
    if len(test_dataset) == 0:
        print("No images in the test set for visualization!")
        return

    # Get first image-label pair from dataset
    img, label = test_dataset[0]
    
    # Add batch dimension and move to device
    img_batch = img.unsqueeze(0).to(device)

    # Load model
    model = WatermarkDetectionMobileNetV3(pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move to device
    model.to(device)
    
    # Set to evalualtion mode
    model.eval()

    # Initialize list to store the output of the hooked layer
    feature_maps = []

    # Define function to capture layer output during forward pass
    def hook_fn(module, input, output):
        feature_maps.append(output)
        
    # Register hook on first convolutional layer
    layer_to_hook = model.base_model.features[0]
    hook_handle = layer_to_hook.register_forward_hook(hook_fn)

    # Disable gradients
    with torch.no_grad():
        # Run single forward pass
        _ = model(img_batch)

    # Remove the hook
    hook_handle.remove()

    # Convert to numpy array
    fmap = feature_maps[0].squeeze().cpu().numpy()
    
    # Limit visualization to first 16 channels
    num_maps = min(fmap.shape[0], 16)
    
    # Visualize up to 16 channels
    plt.figure(figsize=(12, 8))
    for i in range(num_maps):
        plt.subplot(4, 4, i + 1)
        plt.imshow(fmap[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f"Feature Maps from {output_prefix} Model (Conv Layer 0)")
    plt.tight_layout()
    plt.savefig(f"results/{output_prefix}_feature_maps.png") # Save visualization to file