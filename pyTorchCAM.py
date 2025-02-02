import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
 
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
 
        # Placeholders for the gradients and activations
        self.gradients = None
        self.activations = None
 
        # Register hooks
        self._register_hooks()
 
    def _register_hooks(self):
        # Hook for the activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
 
        # Hook for the gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
 
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
 
    def generate_heatmap(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass with specified target
        target = output[0, class_idx]
        target.backward()

        # Get the gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Global average pooling on the gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Compute the heatmap by averaging and applying ReLU
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize the heatmap
        heatmap /= (torch.max(heatmap) + 1e-10)
        heatmap = heatmap - heatmap.min()
        heatmap /= heatmap.max()

        # Further enhance red areas
        heatmap = heatmap.pow(0.3)  # Lower exponent to expand high values

        # Apply larger Gaussian blur for smoother regions
        heatmap = cv2.GaussianBlur(heatmap.cpu().numpy(), (51, 51), 0)

        return heatmap, class_idx
 
# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])
 
def preprocess_image(img_path):
    # Load the image
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return img, input_batch

def overlay_heatmap(heatmap, original_img, alpha, colormap=cv2.COLORMAP_JET):
    # Convert PIL image to OpenCV format
    img = np.array(original_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Normalize and convert heatmap to uint8
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)

    # Intensify red channel
    heatmap_colored[:, :, 2] = np.clip(heatmap_colored[:, :, 2] * 4.0, 0, 255)  # Red
    heatmap_colored[:, :, 1] = np.clip(heatmap_colored[:, :, 1] * 1.0, 0, 255)  # Green
    heatmap_colored[:, :, 0] = np.clip(heatmap_colored[:, :, 0] * 1.0, 0, 255)  # Blue

    # Overlay the heatmap on the image
    overlayed_img = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)

    return overlayed_img

def show_gradcam(img_path, model, target_layer, class_idx=None):
    # Preprocess the image
    original_img, input_batch = preprocess_image(img_path)

    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap, predicted_class = gradcam.generate_heatmap(input_batch, class_idx)

    # Overlay the heatmap on the image
    overlayed_img = overlay_heatmap(heatmap, original_img)

    # Get class labels
    labels = load_imagenet_labels()
    predicted_label = labels[predicted_class]

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.axis('off')
    plt.imshow(original_img)

    plt.subplot(1, 2, 2)
    plt.title(f'Grad-CAM: {predicted_label}')
    plt.axis('off')
    # Convert BGR to RGB for displaying
    plt.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
    plt.show()
 
def load_imagenet_labels():
    # Download ImageNet labels
    import urllib.request
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    import json
    with urllib.request.urlopen(url) as response:
        labels = json.loads(response.read())
    return labels
 
# Updated function to handle device placement
def show_gradcam_with_device(img_path, model, target_layer, device, overlay=False, class_idx=None):
    # Preprocess the image
    original_img, input_batch = preprocess_image(img_path)
    input_batch = input_batch.to(device)
    alphaValue = 0.5
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)
 
    # Generate heatmap
    heatmap, predicted_class = gradcam.generate_heatmap(input_batch, class_idx)

    if (overlay):
        alphaValue = 1
 
    # Overlay the heatmap on the image
    overlayed_img = overlay_heatmap(heatmap, original_img, alphaValue)
 
    # Get class labels
    labels = load_imagenet_labels()
    predicted_label = labels[predicted_class]
 
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.axis('off')
    plt.imshow(original_img)
 
    plt.subplot(1, 2, 2)
    plt.title(f'Grad-CAM: {predicted_label}')
    plt.axis('off')
    # Convert BGR to RGB for displaying
    plt.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
    plt.show()
 
import torch
import torch.nn as nn
 
# Define the same model architecture as before
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Adjusted for 224x224 input
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)  # Binary classification
 
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
 
if __name__ == '__main__':
    # Path to your input image
    img_path = "archive/train_data/ff441adde201401cade2b6f046f4e3f9.jpg"  # Replace with your image path
    #img_path = "archive/train_data/fff35e73dcb54e2d87364d0ed0787db0.jpg"
    # Check if CUDA is available and use GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = CNN()
 
    # Load saved weights
    model.load_state_dict(torch.load('pytorch_ai_image_classifier.pth'))
 
    # Set model to evaluation mode
    model.eval()
    
    # Now you can use the model for inference

    
       
 
    # Select target layer
    target_layer = model.conv2
 
    # Call the function
    def makeHeatMap (img_path, flag=False):
        show_gradcam_with_device(img_path, model, target_layer, device, flag)

    makeHeatMap(img_path, True)
 


