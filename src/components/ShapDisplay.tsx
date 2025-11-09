
import React from 'react';

interface ShapDisplayProps {
  shapAnalysis: string;
}

const ShapFeatureBar: React.FC<{ feature: string; color: string; width: string }> = ({ feature, color, width }) => (
    <div className="flex items-center text-sm mb-2">
        <span className="w-24 text-slate-600 text-right pr-4">{feature}</span>
        <div className="flex-1 bg-slate-200 rounded-full h-5 shadow-inner">
            <div className={`${color} h-5 rounded-full transition-all duration-500 ease-out`} style={{ width }}></div>
        </div>
    </div>
);


export const ShapDisplay: React.FC<ShapDisplayProps> = ({ shapAnalysis }) => {
  return (
    <div className="bg-slate-100 p-4 rounded-lg border border-slate-200">
      <h3 className="text-sm font-medium text-slate-600 mb-2">Interpretability: SHAP Analysis (Metadata)</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
        <div>
            <p className="text-xs text-slate-500 mb-3">
                Simulated feature importance plot. Red bars push the prediction towards Pneumonia, blue bars push it towards Normal.
            </p>
            <div className="p-4 bg-white rounded-lg shadow-sm">
                <ShapFeatureBar feature="Age" color="bg-red-500" width="70%" />
                <ShapFeatureBar feature="View Position" color="bg-red-500" width="45%" />
                <ShapFeatureBar feature="Sex" color="bg-sky-500" width="20%" />
                <div className="text-center text-xs text-slate-400 mt-3 pt-2 border-t">
                    <span className="text-sky-600 font-medium">◄ More "Normal"</span>
                    <span className="mx-4 text-slate-300">|</span>
                    <span className="text-red-600 font-medium">More "Pneumonia" ►</span>
                </div>
            </div>
        </div>
        <div className="prose prose-sm max-w-none">
            <h4 className="text-xs font-semibold mb-1 text-slate-500">AI-Generated Explanation</h4>
            <p className="text-slate-700 bg-white p-3 rounded-md border border-slate-200 shadow-sm">
                {shapAnalysis}
            </p>
        </div>
      </div>
    </div>
  );
};
