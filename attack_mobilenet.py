# Imports
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchattacks import FGSM, PGD

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define function to generate adversarial examples
def adversarial_attack(img: Image.Image, model, attack_type="fgsm", epsilon=0.03, device=device):
    # Resize to model input size and convert to PyTorch tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Apply transform and add batch dimension
    x = transform(img).unsqueeze(0).to(device)
    
    # Create a dummy target label for attack generation
    label = torch.tensor([1], dtype=torch.long).to(device)

    # Choose adversarial attack type based on input argument
    if attack_type == "fgsm":
        atk = FGSM(model, eps=epsilon)
    elif attack_type == "pgd":
        atk = PGD(model, eps=epsilon, alpha=epsilon/3, steps=10)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    # Generate adversarial example
    adv = atk(x, label)
    
    # Remove batch, move, to CPU, and convert to NumPy array
    adv = adv.squeeze().detach().cpu().numpy()
    
    # Clamp to [0, 1]
    adv = np.clip(adv, 0, 1)
    
    # Rescale to [0, 255] and convert to uint8
    adv = (adv * 255).astype(np.uint8)
    
    # Change from CHW to HWC format for PIL
    adv = np.transpose(adv, (1, 2, 0))
    
    # Return as a PIL image
    return Image.fromarray(adv)
