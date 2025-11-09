
import React from 'react';
import { ImageUploader } from './ImageUploader';
import { ClinicalDataForm } from './ClinicalDataForm';
import { ClinicalData, ModelChoice } from '../types';
import { Card } from './ui/Card';
import { Button } from './ui/Button';
import { AnalyzeIcon } from './icons';
import { Toggle } from './ui/Toggle';

interface InputPanelProps {
  imageFile: File | null;
  setImageFile: (file: File | null) => void;
  clinicalData: ClinicalData;
  setClinicalData: (data: ClinicalData) => void;
  onAnalyze: () => void;
  isLoading: boolean;
  modelChoice: ModelChoice;
  setModelChoice: (choice: ModelChoice) => void;
}

export const InputPanel: React.FC<InputPanelProps> = ({
  imageFile,
  setImageFile,
  clinicalData,
  setClinicalData,
  onAnalyze,
  isLoading,
  modelChoice,
  setModelChoice,
}) => {
  return (
    <Card>
      <div className="p-6">
        <h2 className="text-lg font-semibold text-slate-700 mb-4">Input Data</h2>
        <div className="space-y-6">
          <Toggle
            label="Analysis Model"
            option1={{ value: 'gemini', label: 'Gemini Simulation' }}
            option2={{ value: 'local', label: 'Local Model' }}
            value={modelChoice}
            onChange={setModelChoice}
          />
          <ImageUploader file={imageFile} setFile={setImageFile} />
          <ClinicalDataForm data={clinicalData} setData={setClinicalData} />
          <Button
            onClick={onAnalyze}
            disabled={!imageFile || isLoading}
            className="w-full"
          >
            <AnalyzeIcon className="h-5 w-5 mr-2" />
            {isLoading ? 'Analyzing...' : 'Run Analysis'}
          </Button>
        </div>
      </div>
    </Card>
  );
};
