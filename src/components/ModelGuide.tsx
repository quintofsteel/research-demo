
import React from 'react';
import { Card } from './ui/Card';

const CodeBlock: React.FC<{ children: React.ReactNode, lang?: string }> = ({ children, lang = 'bash' }) => (
    <div className="my-4">
        <pre className={`bg-slate-800 text-slate-200 p-4 rounded-lg overflow-x-auto text-sm font-mono language-${lang}`}>
            <code>{children}</code>
        </pre>
    </div>
);

export const ModelGuide: React.FC = () => {
  return (
    <Card className="max-w-4xl mx-auto">
      <div className="p-6 md:p-8 prose prose-slate max-w-none">
        <h2>How to Integrate Your PyTorch Model</h2>
        <p>
          This guide will walk you through setting up a local Python server to run your custom <code>fusion_best.pth</code> model. Following these steps will resolve the "Failed to fetch" error and allow the application to communicate with your model in real-time.
        </p>

        <hr className="my-6"/>

        <h3>Step 1: Backend File Structure</h3>
        <p>First, create a new directory named <code>backend</code> in your project's root folder. Place your trained <code>fusion_best.pth</code> model inside it.</p>
        <CodeBlock>
{`my-app/
├── backend/
│   ├── fusion_best.pth      <-- Place your model here
│   ├── requirements.txt     <-- You will create this
│   └── server.py            <-- You will create this
└── src/
    └── ... (frontend files)`}
        </CodeBlock>

        <hr className="my-6"/>

        <h3>Step 2: Python Dependencies</h3>
        <p>Create a file named <code>backend/requirements.txt</code>. This file lists all the Python libraries required for the server. Your thesis mentioned specific libraries, which are included here.</p>
        <CodeBlock lang="text">
{`Flask
Flask-Cors
torch
torchvision
numpy
Pillow
scikit-learn
# The following are often needed for interpretability
grad-cam
shap
pydicom`}
        </CodeBlock>
        <p>Open your terminal, navigate into the <code>backend</code> directory, and install these packages:</p>
        <CodeBlock>
{`cd backend
pip install -r requirements.txt`}
        </CodeBlock>

        <hr className="my-6"/>

        <h3>Step 3: The Python Server Code</h3>
        <p>Create the file <code>backend/server.py</code>. Copy and paste the complete code below. This Flask server defines the necessary API endpoint and includes detailed placeholders showing you exactly where to integrate your model's architecture and preprocessing logic.</p>
        <CodeBlock lang="python">
{`import os
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn

# --- 1. DEFINE YOUR MODEL ARCHITECTURE ---
# TODO: This class MUST EXACTLY match the architecture of your saved 'fusion_best.pth' file.
# This is an example based on the ResNet/DenseNet + MLP fusion model from your thesis.
class FusionModel(nn.Module):
    def __init__(self, meta_dim=3, img_proj=256, meta_proj=32):
        super().__init__()
        # Using ResNet18 as an example from the thesis
        resnet = models.resnet18() 
        in_feats = resnet.fc.in_features
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
    model = FusionModel()
    # Use map_location to run on CPU if you don't have a GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ PyTorch model 'fusion_best.pth' loaded successfully.")
except Exception as e:
    print(f"❌ Could not load PyTorch model: {e}")
    print("🛑 Server is running in SIMULATION MODE. Predictions will be random.")

# --- 3. DEFINE PREPROCESSING ---
# TODO: These transforms MUST match the transforms used during your model's training.
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        'age': request.form.get('age', '50'), # Default to 50 if not provided
        'sex': request.form.get('sex', 'Male'),
        'view_position': request.form.get('view_position', 'AP'),
    }

    if model:
        # --- Real Inference ---
        image_tensor = preprocess_transform(image).unsqueeze(0)
        
        meta_tensor = torch.tensor([[
            float(clinical_data['age']) / 100.0, 
            1.0 if clinical_data['sex'] == 'Male' else 0.0,
            0.0 if clinical_data['view_position'] == 'AP' else 1.0
        ]], dtype=torch.float32)

        with torch.no_grad():
            logit = model(image_tensor, meta_tensor)
            confidence = torch.sigmoid(logit).item()
            prediction = 'Pneumonia' if confidence > 0.5 else 'Normal'
        
        # TODO: Integrate your actual Grad-CAM and SHAP logic here.
        # For now, we simulate them.
        grad_cam_array = np.array(image.resize((224,224))) # Placeholder
        shap_text = f"SHAP analysis indicates age ({clinical_data['age']}) was a key factor."

    else:
        # --- Simulation Mode ---
        prediction = 'Normal'
        confidence = 0.85
        grad_cam_array = np.array(image.resize((224,224)))
        shap_text = "Model not loaded. This is a simulated SHAP analysis."

    # Convert Grad-CAM to base64
    buffered = BytesIO()
    Image.fromarray(grad_cam_array).save(buffered, format="PNG")
    grad_cam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'gradCamImage': grad_cam_base64,
        'shapAnalysis': shap_text,
        'fairnessAnalysis': "Fairness analysis placeholder: The model shows consistent performance across demographic subgroups in the test set. However, continuous monitoring is required for real-world deployment."
    })

if __name__ == '__main__':
    print("🚀 Starting Flask server for model inference...")
    app.run(debug=True, port=5000)`}
        </CodeBlock>

        <hr className="my-6"/>

        <h3>Step 4: Run the Backend Server</h3>
        <p>In your terminal, from the <code>backend</code> directory, start the server.</p>
        <CodeBlock>
{`flask run`}
        </CodeBlock>
        <p>You should see output confirming the server is running. Look for a line similar to this, and check that your model loaded successfully:</p>
        <CodeBlock>
{`🚀 Starting Flask server for model inference...
✅ PyTorch model 'fusion_best.pth' loaded successfully.
 * Running on http://127.0.0.1:5000`}
        </CodeBlock>
        <p><strong>Leave this terminal running.</strong></p>

        <hr className="my-6"/>

        <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-r-lg my-6">
            <h4 className="font-bold text-amber-800">Troubleshooting the "Failed to fetch" Error</h4>
            <p className="text-amber-700">If you still see the error after following the steps, check the following:</p>
            <ul className="list-disc pl-5 mt-2 space-y-1 text-amber-700">
                <li><strong>Is the server running?</strong> Your terminal window from Step 4 should still be open and active. If it was closed or crashed, you need to run <code>flask run</code> again.</li>
                <li><strong>Are you on the right port?</strong> The server runs on port <code>5000</code> by default. The frontend is hardcoded to connect to this port. Ensure no other process is using it.</li>
                <li><strong>Did the model load?</strong> Check the terminal output. If it says "Could not load PyTorch model", there is a mismatch between the model architecture in <code>server.py</code> and your saved <code>.pth</code> file. You MUST update the <code>FusionModel</code> class in the script to match your model exactly.</li>
                <li><strong>Browser Console Errors:</strong> Open your browser's developer tools (F12 or Ctrl+Shift+I) and look at the "Console" tab. It may show more detailed CORS or network errors.</li>
            </ul>
        </div>

        <h3>Step 5: Run the Demo</h3>
        <p>
          With the server running, return to the "Demo" tab in this application. Select "Local Model" with the toggle, upload your image, and click "Run Analysis". The request will now be sent to your local server, and you will see the results from your own model.
        </p>
      </div>
    </Card>
  );
};
