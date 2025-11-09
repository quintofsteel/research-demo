
import React from 'react';

interface PredictionResultProps {
  prediction: 'Pneumonia' | 'Normal';
  confidence: number;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({ prediction, confidence }) => {
  const isPneumonia = prediction === 'Pneumonia';
  const confidencePercentage = (confidence * 100).toFixed(1);

  const baseClasses = "px-4 py-2 rounded-full text-base font-bold tracking-wide";
  const colorClasses = isPneumonia
    ? "bg-red-100 text-red-800"
    : "bg-green-100 text-green-800";

  return (
    <div className="bg-slate-100 p-4 rounded-lg border border-slate-200">
      <h3 className="text-sm font-medium text-slate-600 mb-3 text-center">Model Prediction</h3>
      <div className="flex items-center justify-around">
        <span className={`${baseClasses} ${colorClasses}`}>{prediction}</span>
        <div className="text-center border-l border-slate-300 pl-6 ml-4">
          <p className="font-bold text-slate-800 text-2xl">{confidencePercentage}%</p>
          <p className="text-xs text-slate-500">Confidence</p>
        </div>
      </div>
    </div>
  );
};
