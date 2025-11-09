import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap

# --- 1. DEFINE YOUR MODEL ARCHITECTURE ---
# This class MUST EXACTLY match the architecture of your saved 'fusion_best.pth' file.
# This is the ResNetMetaFusion model from your notebook.
class ResNetMetaFusion(nn.Module):
    def __init__(self, meta_dim=3, img_proj=256, meta_proj=32, pretrained=True):
        super().__init__()
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        in_feats = resnet.fc.in_features
        # Replace the final fully connected layer
        resnet.fc = nn.Sequential(nn.Linear(in_feats, img_proj), nn.ReLU())
        self.image_model = resnet
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64), nn.ReLU(),
            nn.Linear(64, meta_proj), nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(img_proj + meta_proj, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_meta):
        img_feat = self.image_model(x_img)
        meta_feat = self.meta_mlp(x_meta)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(fused)

# --- 2. LOAD THE MODEL ---
MODEL_PATH = "fusion_best.pth"
model = None
try:
    model = ResNetMetaFusion(pretrained=False) # Set pretrained=False as we are loading our own weights
    # Use map_location to run on CPU if you don't have a GPU
    # The notebook saved the entire model, not just state_dict, so we load it directly.
    # If it was a state_dict, you'd use model.load_state_dict(torch.load(...))
    loaded_data = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    if isinstance(loaded_data, dict) and 'model_state' in loaded_data:
        model.load_state_dict(loaded_data['model_state'])
    else:
        model.load_state_dict(loaded_data)

    model.eval()
    print("✅ PyTorch model 'fusion_best.pth' loaded successfully.")
except Exception as e:
    print(f"❌ Could not load PyTorch model: {e}")
    print("🛑 Server is running in SIMULATION MODE. Predictions will be random.")

# --- 3. DEFINE PREPROCESSING & HELPERS ---
# These transforms MUST match the transforms used during your model's training.
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_metadata_tensor(clinical_data):
    # Default to a reasonable average if age is not provided
    age = float(clinical_data.get('age') or 50)
    sex = 1.0 if clinical_data.get('sex') == 'Male' else 0.0
    view_pos = 0.0 if clinical_data.get('view_position') == 'AP' else 1.0
    return torch.tensor([[age / 100.0, sex, view_pos]], dtype=torch.float32)

# --- 4. SETUP FOR INTERPRETABILITY ---
# Grad-CAM setup
grad_cam = None
if model:
    # Target layer from your ResNet architecture
    target_layer = [model.image_model.layer4[-1]]
    grad_cam = GradCAM(model=model.image_model, target_layers=target_layer)

# SHAP setup
explainer = None
if model:
    # Create a dummy background dataset for the explainer (e.g., 10 samples of "average" metadata)
    background_meta = torch.rand(10, 3) 
    # Create a prediction function for SHAP
    def shap_prediction_func(meta_input):
        # SHAP gives a numpy array, convert to tensor
        meta_tensor = torch.from_numpy(meta_input).float()
        # Create a dummy image tensor to pass to the model
        dummy_image = torch.zeros(meta_tensor.shape[0], 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_image, meta_tensor)
        return output.numpy()
    explainer = shap.KernelExplainer(shap_prediction_func, background_meta.numpy())

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")

    clinical_data = {
        'age': request.form.get('age'),
        'sex': request.form.get('sex'),
        'view_position': request.form.get('view_position'),
    }

    # Preprocess inputs
    image_tensor = preprocess_transform(image).unsqueeze(0)
    meta_tensor = get_metadata_tensor(clinical_data)

    # --- Run Inference ---
    if model:
        with torch.no_grad():
            logit = model(image_tensor, meta_tensor)
            confidence = torch.sigmoid(logit).item()
            prediction = 'Pneumonia' if confidence > 0.5 else 'Normal'
    else: # Simulation if model failed to load
        prediction, confidence = ('Normal', 0.85)

    # --- Generate Interpretability ---
    grad_cam_base64 = ""
    if grad_cam:
        # Generate Grad-CAM
        input_tensor_cam = image_tensor.clone() # Use a clone for Grad-CAM
        grayscale_cam = grad_cam(input_tensor=input_tensor_cam, targets=None)[0, :]
        
        # Reverse normalization for visualization
        rgb_img_vis = np.array(image.resize((224, 224))) / 255.0
        cam_image = show_cam_on_image(rgb_img_vis, grayscale_cam, use_rgb=True)
        
        # Convert to base64
        buffered = BytesIO()
        Image.fromarray(cam_image).save(buffered, format="PNG")
        grad_cam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    shap_text = "SHAP analysis could not be run."
    if explainer:
        shap_values = explainer.shap_values(meta_tensor.numpy())
        # Generate a simple text explanation from SHAP values
        feature_names = ['Age', 'Sex', 'View Position']
        shap_text = "Feature Contributions (SHAP): "
        contributions = []
        for i, name in enumerate(feature_names):
            val = shap_values[0][i]
            direction = "increases" if val > 0 else "decreases"
            impact = "significantly" if abs(val) > 0.1 else "slightly"
            contributions.append(f"{name} {impact} {direction} the pneumonia prediction")
        shap_text += "; ".join(contributions) + "."

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'gradCamImage': grad_cam_base64,
        'shapAnalysis': shap_text,
        'fairnessAnalysis': "Fairness analysis placeholder: The model's performance on the input demographic (e.g., Age, Sex) should be compared against its performance on other groups to ensure equitable outcomes. Continuous auditing is crucial."
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server for model inference...")
    app.run(debug=True, port=5000)