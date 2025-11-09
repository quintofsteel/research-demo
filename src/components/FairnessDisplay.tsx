
import React from 'react';
import { ShieldCheckIcon } from './icons';

interface FairnessDisplayProps {
  fairnessAnalysis: string;
}

export const FairnessDisplay: React.FC<FairnessDisplayProps> = ({ fairnessAnalysis }) => {
  return (
    <div className="bg-slate-100 p-4 rounded-lg border border-slate-200">
      <h3 className="text-sm font-medium text-slate-600 mb-2 flex items-center">
        <ShieldCheckIcon className="h-5 w-5 mr-2 text-sky-600" />
        Fairness & Bias Considerations
      </h3>
      <div className="prose prose-sm max-w-none">
        <p className="text-slate-700 bg-white p-3 rounded-md border border-slate-200 shadow-sm">
            {fairnessAnalysis}
        </p>
        <p className="text-xs text-slate-500 mt-2">
            <strong>Note:</strong> This is an AI-generated analysis based on the principles outlined in the thesis regarding model fairness, generalization, and performance across different demographic subgroups.
        </p>
      </div>
    </div>
  );
};
