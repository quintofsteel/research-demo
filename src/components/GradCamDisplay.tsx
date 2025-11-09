
import React from 'react';

interface GradCamDisplayProps {
  originalImage: string | null;
  gradCamImage: string | null;
}

export const GradCamDisplay: React.FC<GradCamDisplayProps> = ({ originalImage, gradCamImage }) => {
  return (
    <div className="bg-slate-100 p-4 rounded-lg border border-slate-200">
      <h3 className="text-sm font-medium text-slate-600 mb-2">Interpretability: Grad-CAM</h3>
      <p className="text-xs text-slate-500 mb-3">
        Simulated heatmap showing areas the model focused on for its prediction.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="text-center">
          <h4 className="text-xs font-semibold mb-2 text-slate-500">Original X-Ray</h4>
          {originalImage ? (
            <img src={originalImage} alt="Original X-ray" className="rounded-md w-full shadow" />
          ) : (
            <div className="w-full h-48 bg-slate-200 rounded-md flex items-center justify-center text-slate-500">No Image</div>
          )}
        </div>
        <div className="text-center">
          <h4 className="text-xs font-semibold mb-2 text-slate-500">Simulated Grad-CAM</h4>
          {gradCamImage ? (
            <img src={gradCamImage} alt="Grad-CAM" className="rounded-md w-full shadow" />
          ) : (
            <div className="w-full h-48 bg-slate-200 rounded-md flex items-center justify-center text-slate-500">Not Generated</div>
          )}
        </div>
      </div>
    </div>
  );
};
