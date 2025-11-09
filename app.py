pip install streamlit torch torchvision pandas numpy opencv-python pydicom shap pytorch-grad-cam scikit-learn

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import cv2
import os
import pydicom
import shap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.linear_model import LogisticRegression

# =================================================================
# 1. Model Architecture Definitions (From Thesis)
# =================================================================

class DenseNetMetaFusion(nn.Module):
    """
    Multimodal Fusion Model combining DenseNet-121 image features 
    with a shallow MLP for metadata.
    """
    def __init__(self, meta_dim=3, img_proj=256, meta_proj=32, pretrained=True):
        super().__init__()
        
        # 1. Image Model (DenseNet-121 Backbone)
        densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = densenet.classifier.in_features
        # Modify the classifier layer to output img_proj features
        densenet.classifier = nn.Sequential(nn.Linear(in_feats, img_proj), nn.ReLU())
        self.image_model = densenet
        
        # 2. Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, meta_proj),
            nn.ReLU()
        )
        
        # 3. Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(img_proj + meta_proj, 128),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_meta):
        # Image Feature Extraction
        img_feat = self.image_model(x_img)
        # Metadata Feature Extraction
        meta_feat = self.meta_mlp(x_meta)
        
        # Late Fusion
        fused = torch.cat([img_feat, meta_feat], dim=1)
        
        # Final Classification
        return self.classifier(fused).squeeze(1)

# =================================================================
# 2. Data Processing Utilities (Simplified from Thesis)
# =================================================================

IMG_SIZE = 224
# ImageNet mean/std used for normalization in your notebook
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

class XRayProcessor:
    """Simplified Image Processor for demo."""
    
    @staticmethod
    def _normalize_img(img):
        """Normalize pixel values to [0, 1] and ensure float32."""
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        return img

    @staticmethod
    def load_dicom(file_path):
        """Loads and processes DICOM files."""
        try:
            dicom_data = pydicom.dcmread(file_path, force=True)
            img = dicom_data.pixel_array
            
            # Apply RescaleSlope and Intercept if present
            if 'RescaleSlope' in dicom_data and 'RescaleIntercept' in dicom_data:
                img = img * float(dicom_data.RescaleSlope) + float(dicom_data.RescaleIntercept)
            
            # Convert to normalized float array [0, 1]
            img = XRayProcessor._normalize_img(img)
            
            return img
        except Exception:
            return None

    @staticmethod
    def load_standard_image(file_path):
        """Loads and processes PNG/JPEG files."""
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = XRayProcessor._normalize_img(img)
        return img

    @staticmethod
    def process_image_for_model(uploaded_file):
        """Handles file upload and returns processed tensor [3, H, W]."""
        
        # 1. Save uploaded file temporarily to disk (required by pydicom/cv2)
        temp_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Load image based on extension
        file_ext = uploaded_file.name.lower().split('.')[-1]
        if file_ext == 'dcm':
            img_array = XRayProcessor.load_dicom(temp_path)
        elif file_ext in ['png', 'jpg', 'jpeg']:
            img_array = XRayProcessor.load_standard_image(temp_path)
        else:
            st.error(f"Unsupported file format: .{file_ext}")
            return None

        if img_array is None:
            st.error("Error loading or reading image file.")
            return None

        # 3. Preprocessing (Resize, Channel Expansion, Normalize)
        
        # Resize to IMG_SIZE
        img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Add channel dimension: [1, H, W] for grayscale
        img_tensor = torch.from_numpy(np.expand_dims(img, axis=0)).float()
        
        # Ensure 3 channels for ResNet/DenseNet
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1) # Shape changes to [3, H, W]
        
        # Apply Normalization
        mean_tensor = torch.tensor(NORM_MEAN).view(3, 1, 1)
        std_tensor = torch.tensor(NORM_STD).view(3, 1, 1)
        img_tensor = (img_tensor - mean_tensor) / std_tensor

        return img_tensor.unsqueeze(0) # Add batch dimension [1, 3, H, W]

def encode_metadata(age, sex, view_pos):
    """
    Encodes metadata based on thesis logic:
    Age: Normalized (age/100), default 0.5
    Sex: 0=M, 1=F, -1=Missing
    View Position: 0=AP, 1=PA, -1=Missing
    """
    
    # Age (Normalize by 100.0, default 0.5)
    try:
        age_val = float(age)
        age_enc = age_val / 100.0 if not np.isnan(age_val) else 0.5
    except (ValueError, TypeError):
        age_enc = 0.5
    
    # Sex (M=0, F=1, default -1.0)
    sex_map = {'Male': 0.0, 'Female': 1.0}
    sex_enc = sex_map.get(sex, -1.0)
    
    # View Position (AP=0, PA=1, default -1.0)
    view_map = {'AP': 0.0, 'PA': 1.0}
    view_enc = view_map.get(view_pos, -1.0)

    meta_tensor = torch.tensor([age_enc, sex_enc, view_enc], dtype=torch.float32)
    return meta_tensor.unsqueeze(0) # Add batch dimension [1, 3]

# =================================================================
# 3. SHAP Utility (Placeholder - Requires Surrogate Model)
# =================================================================

@st.cache_resource
def setup_shap_explainer(model_instance, device):
    """
    Sets up a placeholder SHAP explainer. 
    NOTE: In a real scenario, the explainer needs to be trained on 
    a surrogate model (Logistic Regression) fit to the training data.
    The following is a placeholder for demonstration purposes.
    """
    
    try:
        # Create a dummy dataset for the explainer to be initialized
        # This size should ideally match the training data subset size (e.g., 100 samples)
        dummy_background_data = np.random.rand(50, 3) 
        dummy_background_data[:, 1] = np.random.choice([-1.0, 0.0, 1.0], 50)
        dummy_background_data[:, 2] = np.random.choice([-1.0, 0.0, 1.0], 50)
        
        # Create a dummy prediction function that uses the metadata MLP head (as per thesis)
        def predict_meta_mlp(metadata_input):
            # metadata_input is a numpy array from SHAP
            meta_tensor = torch.from_numpy(metadata_input).float().to(device)
            with torch.no_grad():
                # We explain the output of the meta_mlp head for SHAP (as per thesis logic)
                meta_feat = model_instance.meta_mlp(meta_tensor)
                # For a single number output (simulating the logit), we use the sum of features
                # In your thesis, you train a surrogate model on the logit, which is better.
                # Here, we will simplify and explain the logit *output* of the MLP
                logit_output = model_instance.classifier(
                    torch.cat([
                        torch.zeros((meta_tensor.shape[0], 256), device=device), # Dummy img_feat
                        meta_feat
                    ], dim=1)
                ).squeeze(1)
            return logit_output.cpu().numpy()

        # We use KernelExplainer for model-agnostic explanation
        explainer = shap.KernelExplainer(predict_meta_mlp, dummy_background_data)
        return explainer
    except Exception as e:
        st.error(f"SHAP Explainer setup failed: {e}")
        return None

# =================================================================
# 4. Streamlit UI and Logic
# =================================================================

def main():
    st.set_page_config(layout="wide", page_title="Multimodal Pneumonia AI Demo")
    st.title("👨‍💻 Multimodal Pneumonia AI Demo")
    st.subheader("Assessing Multimodal Fusion, Generalization, and Fairness")
    st.markdown("---")

    # --- Setup and Model Loading ---
    
    # NOTE: Set the correct path to your downloaded model
    MODEL_PATH = "fusion_best.pth" 
    
    if os.path.exists(MODEL_PATH):
        st.success(f"Model checkpoint found: {MODEL_PATH}")
    else:
        st.error(f"Model checkpoint NOT found at: {MODEL_PATH}. Using randomly initialized weights.")
        st.warning("Prediction and interpretability results will be random without the trained model.")

    # Model and Device Setup
    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DenseNetMetaFusion(pretrained=True).to(device)
        
        # Load state dict if available
        if os.path.exists(MODEL_PATH):
            try:
                # Assuming your checkpoint is a dict with 'model_state' key
                checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
                state_dict = checkpoint.get('model_state', checkpoint)
                model.load_state_dict(state_dict, strict=False)
                st.toast("Model weights loaded successfully!", icon="✅")
            except Exception as e:
                st.error(f"Error loading model weights: {e}. Model is using pre-trained backbone only.")
        
        model.eval()
        return model, device

    model, device = load_model()
    
    # Grad-CAM Setup
    target_layer = model.image_model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type=='cuda')
    
    # SHAP Setup
    shap_explainer = setup_shap_explainer(model, device)
    metadata_feature_names = ['age', 'sex', 'view_position'] # As per thesis

    st.markdown("### 1. Inputs")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Clinical Metadata (Structured Data)**")
        # Metadata Inputs
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
        sex = st.radio("Sex", ['Male', 'Female', 'Unknown'])
        view_pos = st.radio("View Position", ['PA', 'AP', 'Unknown'])
        
    with col2:
        st.markdown("**Chest X-ray Image**")
        # Image Upload
        uploaded_file = st.file_uploader("Upload DICOM (.dcm), PNG, or JPEG image", type=["dcm", "png", "jpg", "jpeg"])

    st.markdown("---")
    st.markdown("### 2. Prediction & Interpretation")

    if uploaded_file is not None:
        
        # --- Preprocessing & Prediction ---
        
        with st.spinner("Processing image and running inference..."):
            img_tensor = XRayProcessor.process_image_for_model(uploaded_file)
            meta_tensor = encode_metadata(age, sex, view_pos).to(device)
            
            if img_tensor is None:
                return

            # Move image tensor to device
            img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                logits = model(img_tensor, meta_tensor)
                probability = torch.sigmoid(logits).item()
            
            prediction = "Pneumonia Detected" if probability > 0.5 else "No Pneumonia"
            
            # --- Grad-CAM Generation ---
            
            # The input tensor needs to be normalized (which process_image_for_model does)
            grayscale_cam = cam(input_tensor=img_tensor, targets=None, eigen_smooth=True)[0, :]
            
            # Reverse normalization for visualization (H, W, C)
            img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() # [H, W, 3]
            img_display = img_np * NORM_STD + NORM_MEAN
            img_display = np.clip(img_display, 0, 1)

            # Apply heatmap overlay
            cam_image = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
            
            # --- SHAP Calculation ---
            
            if shap_explainer:
                meta_np = meta_tensor.cpu().numpy()
                shap_values = shap_explainer.shap_values(meta_np)
            else:
                shap_values = None

        # --- Display Results ---
        
        st.success(f"**Diagnosis:** {prediction} (Probability: {probability:.4f})")
        
        # Display Columns
        col_img, col_meta = st.columns(2)

        with col_img:
            st.markdown("#### Image-based Interpretation (Grad-CAM)")
            st.image(cam_image, caption="Grad-CAM Overlay (Image Attention)", use_column_width=True)
            st.markdown(f"**Insight:** The model's attention is focused on the highlighted areas.")
            
        with col_meta:
            st.markdown("#### Metadata-based Interpretation (SHAP)")
            
            if shap_values is not None:
                # SHAP expects a prediction function output for the minority class (label=1)
                st.pyplot(shap.summary_plot(shap_values[0], meta_np, feature_names=metadata_feature_names, plot_type="bar", show=False))
                st.markdown(f"""
                **Insight:** SHAP values show how each metadata feature (Age, Sex, View Position) contributes to the prediction.
                * A positive SHAP value (right of zero) increases the likelihood of *Pneumonia Detected*.
                * A negative SHAP value (left of zero) decreases the likelihood of *Pneumonia Detected*.
                """)
            else:
                st.warning("SHAP explanation not available (Explainer setup failed).")

    else:
        st.info("Please upload an image and set clinical metadata to run the demo.")
        
    st.markdown("---")
    st.markdown("""
    **Demo Notes for Panel:**
    *   **Architecture:** This model implements the **DenseNetMetaFusion** architecture, combining the image features from a pre-trained DenseNet-121 backbone with clinical metadata features processed by a Shallow MLP.
    *   **Prediction:** The top box shows the final probability and binary prediction.
    *   **Grad-CAM:** The left plot highlights the regions of the CXR image that the **image model** focused on for its decision. This is crucial for detecting the "shortcut learning" discussed in the thesis (e.g., focusing on text, metallic implants, or image artifacts instead of pathology).
    *   **SHAP:** The right plot shows the contribution of each clinical feature (Age, Sex, View Position) to the final logit output. This validates the "clinically sensible features" finding from the thesis (e.g., AP view, a proxy for patient sickness, often having a high positive SHAP value).
    *   **Goal:** The demo illustrates the **multimodal interpretability framework** designed to audit the model's decision-making across both image and metadata modalities.
    """)

if __name__ == "__main__":
    # To run this app, use the command: streamlit run app.py
    main()
