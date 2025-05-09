# Imports
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mobilenet_model import WatermarkDetectionMobileNetV3
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# Define an extended ImageFolder class that also returns the file path
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # Load item
        original_tuple = super().__getitem__(index)
        
        # Get file path
        path = self.imgs[index][0]
        
        return original_tuple + (path,)

# Define function to run model on entire dataset and get predictions, labels, and file paths
def run_inference_with_filenames(model, loader, device):
    # Initialize lists to store results
    y_true, y_pred, filenames = [], [], []
    
    # Disable gradients
    with torch.no_grad():
        # Batch over dataset
        for images, labels, paths in loader:
            # Move to device
            images, labels = images.to(device), labels.to(device)
            
            # Perform forward pass
            outputs = model(images)
            
            # Get predicted class indices
            _, preds = torch.max(outputs, 1)
            
            # Store ground truth label
            y_true.extend(labels.cpu().numpy())
            
            # Store predictions
            y_pred.extend(preds.cpu().numpy())
            
            # Store filenames
            filenames.extend(paths)
            
    return y_true, y_pred, filenames

# Define function to load model and dataset, run inference, calculate metrics, and save results
def evaluate_model(model_path, test_dir, batch_size, output_prefix="experimental"):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define image preproccesing steps
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize image to expected input size
        transforms.ToTensor() # Convert to tensor
    ])
    
    # Load dataset using extended ImageFolder class
    test_dataset = ImageFolderWithPaths(test_dir, transform=transform)
    
    # Create DataLoader
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Get list of classes
    class_names  = test_dataset.classes

    # Load trained watermark detection model
    model = WatermarkDetectionMobileNetV3(pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move to device
    model.to(device)
    
    # Set to evaluation mode
    model.eval()

    # Run inference on test set
    y_true, y_pred, filenames = run_inference_with_filenames(model, test_loader, device)

    # Calculate metrics
    acc  = accuracy_score(y_true, y_pred) # Accuacy
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) # Weighted precision
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0) # Weighted recall
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0) # Weighted F1 score
    cm   = confusion_matrix(y_true, y_pred) # Confusion matrix

    # Print metrics and classification report for validation
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(prec))
    print("Recall: {:.4f}".format(rec))
    print("F1 Score: {:.4f}".format(f1))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Make sure results directory exists to save results to
    os.makedirs("results", exist_ok=True)

    # Convert to dataframe
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Save confusion matrix to CSV
    cm_df.to_csv(f"results/{output_prefix}_confusion_matrix.csv")

    # Create a report dictionary
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Convert to dataframe
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report to CSV
    report_df.to_csv(f"results/{output_prefix}_classification_report.csv")

    # Save summary of metrics
    summary_df = pd.DataFrame([{
        "Model": output_prefix,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }])
    summary_path = f"results/summary_metrics.csv"
    if os.path.exists(summary_path):
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)

    # Get tamper type from filename
    tamper_types = [os.path.basename(path).split("_")[0] for path in filenames]
    
    # Initialize lists to save grouped results
    results_by_tamper = defaultdict(lambda: {"y_true": [], "y_pred": []})
    
    # Group predictions by tamper type
    for yt, yp, tt in zip(y_true, y_pred, tamper_types):
        results_by_tamper[tt]["y_true"].append(yt)
        results_by_tamper[tt]["y_pred"].append(yp)

    # Initialize list to store tamper type-specific metrics
    tamper_eval = []
    
    # Evaluate performance by tamper type
    for tt, res in results_by_tamper.items():
        t_acc = accuracy_score(res["y_true"], res["y_pred"])
        t_prec = precision_score(res["y_true"], res["y_pred"], average='weighted', zero_division=0)
        t_rec = recall_score(res["y_true"], res["y_pred"], average='weighted', zero_division=0)
        t_f1 = f1_score(res["y_true"], res["y_pred"], average='weighted', zero_division=0)
        t_cm = confusion_matrix(res["y_true"], res["y_pred"])
        tamper_eval.append({
            "Tamper Type": tt,
            "Accuracy": t_acc,
            "Precision": t_prec,
            "Recall": t_rec,
            "F1 Score": t_f1
        })
        
        # Save confusion matrix for each tamper type
        plt.figure(figsize=(5, 4))
        sns.heatmap(t_cm, annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion Matrix: {tt}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"results/{output_prefix}_confmat_{tt}.png")
        plt.close()

    # Save tamper type-specific metrics as CSV
    df_tamper = pd.DataFrame(tamper_eval)
    df_tamper.to_csv(f"results/{output_prefix}_tamper_type_metrics.csv", index=False)

    # Plot the worst performing tamper types
    hardest = df_tamper.sort_values("Accuracy").head(5) # Sort by lowest accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(data=hardest, x="Tamper Type", y="Accuracy", palette="coolwarm")
    plt.title("Lowest Accuracy Tamper Types")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"results/{output_prefix}_worst_tamper_types.png")

    # Plot a summary of the general metrics
    plt.figure(figsize=(12, 5))
    
    # Subplot a confusion matrix heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Subplot a bar chart of general metrics
    plt.subplot(1, 2, 2)
    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1}
    plt.bar(metrics.keys(), metrics.values(), color="gray")
    plt.ylim([0, 1])
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.tight_layout()

    # Save composite evaluation plot
    plt.savefig(f"results/{output_prefix}_evaluation_metrics.png")
    print(f"Results saved in 'results/' with prefix '{output_prefix}_'.")
