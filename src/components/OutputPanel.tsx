
import React, { useState } from 'react';
import { AnalysisResult } from '../types';
import { PredictionResult } from './PredictionResult';
import { GradCamDisplay } from './GradCamDisplay';
import { ShapDisplay } from './ShapDisplay';
import { Card } from './ui/Card';
import { Spinner } from './ui/Spinner';
import { InfoIcon, BrainIcon, EyeIcon, BarChartIcon, ShieldCheckIcon } from './icons';
import { FairnessDisplay } from './FairnessDisplay';

interface OutputPanelProps {
  result: AnalysisResult | null;
  isLoading: boolean;
  error: string | null;
  originalImage: string | null;
}

type OutputTab = 'interpret' | 'fairness';

const loadingMessages = [
    "Preparing data and loading model...",
    "Running inference on the provided data...",
    "Generating Grad-CAM and SHAP explanations...",
    "Analyzing potential fairness considerations...",
    "Finalizing results...",
];

const OutputTabButton: React.FC<{tabName: OutputTab, currentTab: OutputTab, onClick: (tab: OutputTab) => void, children: React.ReactNode, icon: React.ReactNode}> = ({ tabName, currentTab, onClick, children, icon }) => (
    <button
      onClick={() => onClick(tabName)}
      className={`flex items-center space-x-2 px-3 py-2 font-medium text-sm rounded-md ${
        currentTab === tabName
          ? 'bg-slate-200 text-slate-800'
          : 'text-slate-500 hover:text-slate-700 hover:bg-slate-100'
      }`}
    >
      {icon}
      <span>{children}</span>
    </button>
);


export const OutputPanel: React.FC<OutputPanelProps> = ({ result, isLoading, error, originalImage }) => {
  const [loadingMessage, setLoadingMessage] = React.useState(loadingMessages[0]);
  const [activeTab, setActiveTab] = useState<OutputTab>('interpret');

  React.useEffect(() => {
    if (isLoading) {
      let i = 0;
      const interval = setInterval(() => {
        i = (i + 1) % loadingMessages.length;
        setLoadingMessage(loadingMessages[i]);
      }, 2500);
      return () => clearInterval(interval);
    }
  }, [isLoading]);

  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full min-h-[400px]">
          <Spinner />
          <p className="mt-4 text-slate-600 text-center w-3/4">{loadingMessage}</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-red-700 bg-red-50 p-4 rounded-lg border border-red-200">
          <h3 className="font-semibold mb-2 text-lg">Analysis Failed</h3>
          <p className="text-center text-sm">{error}</p>
        </div>
      );
    }

    if (result) {
      return (
        <>
          <PredictionResult prediction={result.prediction} confidence={result.confidence} />
          <div className="mt-6">
            <div className="mb-4 border-b border-slate-200">
                <nav className="flex space-x-2" aria-label="Tabs">
                    <OutputTabButton tabName="interpret" currentTab={activeTab} onClick={setActiveTab} icon={<BrainIcon className="h-4 w-4"/>}>Interpretability</OutputTabButton>
                    <OutputTabButton tabName="fairness" currentTab={activeTab} onClick={setActiveTab} icon={<ShieldCheckIcon className="h-4 w-4"/>}>Fairness & Bias</OutputTabButton>
                </nav>
            </div>
            {activeTab === 'interpret' && (
                <div className="space-y-6">
                    <GradCamDisplay originalImage={originalImage} gradCamImage={result.gradCamImage} />
                    <ShapDisplay shapAnalysis={result.shapAnalysis} />
                </div>
            )}
            {activeTab === 'fairness' && (
                <FairnessDisplay fairnessAnalysis={result.fairnessAnalysis} />
            )}
          </div>
        </>
      );
    }

    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[400px] text-slate-500 bg-slate-50 rounded-lg border-2 border-dashed border-slate-300">
        <InfoIcon className="h-12 w-12 mb-4" />
        <h3 className="font-semibold text-slate-700">Awaiting Input</h3>
        <p className="text-center text-sm">Upload an image to begin the analysis.</p>
      </div>
    );
  };

  return (
    <Card>
      <div className="p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Analysis Output</h2>
        {renderContent()}
      </div>
    </Card>
  );
};
