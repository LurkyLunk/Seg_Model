import torch
import torch.onnx
from unet_module import UNet  # Importing the UNet model from unet_module.py

# Number of classes your model predicts
num_classes = 10  # Replace with the actual number of classes

# Instantiate your model
model = UNet(n_class=num_classes)

# Load the trained model weights
model_weights_path = "path_to_your_saved_model/unet_model_epoch_x.pth"  # Replace with the path to your model weights file
model.load_state_dict(torch.load(model_weights_path))

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor matching the input size your model expects
# Adjust the size according to your model's input dimensions
dummy_input = torch.randn(1, 3, 256, 256)  # Example for a single image of size 256x256 with 3 color channels

# Export the model to ONNX format
onnx_model_path = "unet_model.onnx"  # You can change this path if you want to save the ONNX model somewhere specific
torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f"Model exported to ONNX format at: {onnx_model_path}")
