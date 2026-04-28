import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.model import RetinaNet
from PIL import Image
from torchvision import transforms

def generate_heatmap(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    # Target the last layer of EfficientNet [cite: 81]
    target_layers = [model.backbone.conv_head]
    
    # Load and preprocess image
    rgb_img = np.array(Image.open(image_path).convert("RGB")) / 255.0
    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    # Generate CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite("outputs/gradcam/result.jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return visualization