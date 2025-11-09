
import React, { useState, useCallback } from 'react';
import { Header } from './components/Header';
import { InputPanel } from './components/InputPanel';
import { OutputPanel } from './components/OutputPanel';
import { usePneumoniaDetection } from './hooks/usePneumoniaDetection';
import { ClinicalData, AnalysisResult, ModelChoice } from './types';
import { Footer } from './components/Footer';
import { ModelGuide } from './components/ModelGuide';
import { DemoIcon, GuideIcon } from './components/icons';

const App: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [clinicalData, setClinicalData] = useState<ClinicalData>({
    age: '',
    sex: 'Male',
    view_position: 'AP',
  });
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [modelChoice, setModelChoice] = useState<ModelChoice>('gemini');
  const [activeTab, setActiveTab] = useState<'demo' | 'guide'>('demo');
  
  const { analyze, isLoading, error } = usePneumoniaDetection();

  const handleAnalyze = useCallback(async () => {
    if (!imageFile) {
      alert('Please upload a chest X-ray image.');
      return;
    }

    setAnalysisResult(null);
    const result = await analyze(imageFile, clinicalData, modelChoice);
    setAnalysisResult(result);
  }, [imageFile, clinicalData, analyze, modelChoice]);

  const TabButton: React.FC<{tabName: 'demo' | 'guide', currentTab: 'demo' | 'guide', children: React.ReactNode, icon: React.ReactNode}> = ({ tabName, currentTab, children, icon }) => (
    <button
      onClick={() => setActiveTab(tabName)}
      className={`flex items-center space-x-2 px-4 py-2 font-medium text-sm rounded-t-lg border-b-2 transition-colors ${
        currentTab === tabName
          ? 'border-sky-500 text-sky-600 bg-slate-100'
          : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
      }`}
      aria-current={currentTab === tabName ? 'page' : undefined}
    >
      {icon}
      <span>{children}</span>
    </button>
  );

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header />
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="border-b border-slate-200">
          <nav className="-mb-px flex space-x-4" aria-label="Tabs">
            <TabButton tabName="demo" currentTab={activeTab} icon={<DemoIcon className="h-5 w-5" />}>Demo</TabButton>
            <TabButton tabName="guide" currentTab={activeTab} icon={<GuideIcon className="h-5 w-5" />}>Model Integration Guide</TabButton>
          </nav>
        </div>

        <div className="mt-8">
          {activeTab === 'demo' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <InputPanel
                imageFile={imageFile}
                setImageFile={setImageFile}
                clinicalData={clinicalData}
                setClinicalData={setClinicalData}
                onAnalyze={handleAnalyze}
                isLoading={isLoading}
                modelChoice={modelChoice}
                setModelChoice={setModelChoice}
              />
              <OutputPanel
                result={analysisResult}
                isLoading={isLoading}
                error={error}
                originalImage={imageFile ? URL.createObjectURL(imageFile) : null}
              />
            </div>
          )}
          {activeTab === 'guide' && <ModelGuide />}
        </div>
      </main>
      <Footer />
    </div>
  );
};

export default App;
