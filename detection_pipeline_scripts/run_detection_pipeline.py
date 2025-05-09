# Imports
from train_baseline import train_baseline
from train_experimental import train_experimental
from train_finetuned import train_finetuned
from evaluate_model import evaluate_model
from visualize_feature_maps import visualize_feature_maps

# Define baseline model directories and parameters
baseline_train_dir = "custom_dataset/baseline_train"
baseline_val_dir = "custom_dataset/baseline_val"
baseline_model_path = "models/baseline_model.pth"
baseline_epochs = 40
baseline_batch_size = 32
baseline_lr = 1e-4

# Define experimetnal model directories and parameters
experimental_train_dir = "custom_dataset/experimental_train"
experimental_val_dir = "custom_dataset/experimental_val"
experimental_model_path = "models/experimental_model.pth"
experimental_epochs = 40
experimental_batch_size = 32
experimental_lr = 1e-4

# Define finetuned model directory and parameters
finetuned_model_path = "models/finetuned_model.pth"
finetuned_epochs = 40
finetuned_batch_size = 32
finetuned_lr = 1e-4

# Define evaluation dataset paths and parameters
test_dir = "custom_dataset/experimental_test"
unseen_test_dir = "unseen_dataset/unseen_test_combined"
eval_batch_size = 32

# Train baseline model
print("Training baseline model...") # Print status update
train_baseline(
    train_dir=baseline_train_dir,
    val_dir=baseline_val_dir,
    model_save_path=baseline_model_path,
    epochs=baseline_epochs,
    batch_size=baseline_batch_size,
    lr=baseline_lr
)

# Train experimental model
print("Training experimental model...") # Print status update
train_experimental(
    train_dir=experimental_train_dir,
    val_dir=experimental_val_dir,
    model_save_path=experimental_model_path,
    epochs=experimental_epochs,
    batch_size=experimental_batch_size,
    lr=experimental_lr
)

# Finetune baseline model on experimental data
print("Finetuning baseline model...") # Print status update
train_finetuned(
    train_dir=experimental_train_dir,
    val_dir=experimental_val_dir,
    pretrained_model_path=baseline_model_path,
    model_save_path=finetuned_model_path,
    epochs=finetuned_epochs,
    batch_size=finetuned_batch_size,
    lr=finetuned_lr
)

# Evaluate baseline model on experimental test set
print("Evaluating baseline model...") # Print status update
evaluate_model(
    model_path=baseline_model_path,
    test_dir=test_dir,
    batch_size=eval_batch_size,
    output_prefix="baseline"
)

# Evaluate experimental model on experimental test set
print("Evaluating experimental model...") # Print status update
evaluate_model(
    model_path=experimental_model_path,
    test_dir=test_dir,
    batch_size=eval_batch_size,
    output_prefix="experimental"
)

# Evaluate finetuned model on experimental test set
print("Evaluating finetuned model...") # Print status update
evaluate_model(
    model_path=finetuned_model_path,
    test_dir=test_dir,
    batch_size=eval_batch_size,
    output_prefix="finetuned"
)

# Evaluate baseline model on unseen test set
print("Evaluate baseline model on unseen test set...") # Print status update
evaluate_model(
    model_path=baseline_model_path,
    test_dir=unseen_test_dir,
    batch_size=eval_batch_size,
    output_prefix="unseen_baseline"
)

# Evaluate experimental model on unseen test set
print("Evaluate experimental model on unseen test set...") # Print status update
evaluate_model(
    model_path=experimental_model_path,
    test_dir=unseen_test_dir,
    batch_size=eval_batch_size,
    output_prefix="unseen_experimental"
)

# Evaluate finetuned model on unseen test set
print("Evaluate finetuned model on unseen test set...") # Print status update
evaluate_model(
    model_path=finetuned_model_path,
    test_dir=unseen_test_dir,
    batch_size=eval_batch_size,
    output_prefix="unseen_finetuned"
)

# Generate feature maps for baseline model
print("Generating feature Maps for baseline model...") # Print status update
visualize_feature_maps(baseline_model_path, test_dir, output_prefix="baseline")

# Generate feature maps for experimental model
print("Generating feature Maps for experimental model...") # Print status update
visualize_feature_maps(experimental_model_path, test_dir, output_prefix="experimental")

# Generate feature maps for finetuned model
print("Generating feature Maps for finetuned model...") # Print status update
visualize_feature_maps(finetuned_model_path, test_dir, output_prefix="finetuned")

print("Detection pipeline finished!") # Print status update