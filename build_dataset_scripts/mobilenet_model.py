# Imports
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


# Define class for watermark detection model using MobileNetV3-Small
class WatermarkDetectionMobileNetV3(nn.Module):
     def __init__(self, pretrained=True):
        # Initialize parent nn.Module class
        super().__init__()
        
        # Load pretrained MobileNetV3-Small model
        self.base_model = mobilenet_v3_small(pretrained=pretrained)

        # Get input size to first linear layer in original classifier
        in_features = self.base_model.classifier[0].in_features

        # Replace classifier block with custom binary classification head
        self.base_model.classifier = nn.Sequential(
            # Add dense layer projecting to 256-dim feature space
            nn.Linear(in_features, 256),
            
            # Add activation function
            nn.ReLU(inplace=True),
            
            # Add dropout for regularization
            nn.Dropout(p=0.2),
            
            # Add final output layer for 2 classes
            nn.Linear(256, 2)
        )
        
    # Define function to forward pass through model
    def forward(self, x):
        return self.base_model(x)
    
    # Define function to save model weights
    def save(self, path="models/mobilenet_watermark_detector.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(self.state_dict(), path)
        
        # Print status update
        print(f"Model weights saved to {path}")

    # Define function to load model weights and map to device
    def load(self, path, device='cpu', strict=True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.load_state_dict(torch.load(path, map_location=device), strict=strict)
        
        self.to(device)
        
        # Print status update
        print(f"Model weights loaded from {path}")
