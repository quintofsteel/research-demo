
import { useState, useCallback } from 'react';
import { getPneumoniaAnalysis as getGeminiAnalysis } from '../services/geminiService';
import { getLocalModelAnalysis } from '../services/localModelService';
import { ClinicalData, AnalysisResult, ModelChoice } from '../types';

export const usePneumoniaDetection = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyze = useCallback(async (imageFile: File, clinicalData: ClinicalData, modelChoice: ModelChoice): Promise<AnalysisResult | null> => {
    setIsLoading(true);
    setError(null);
    try {
      let result;
      if (modelChoice === 'local') {
        result = await getLocalModelAnalysis(imageFile, clinicalData);
      } else {
        result = await getGeminiAnalysis(imageFile, clinicalData);
      }
      setIsLoading(false);
      return result;
    } catch (err: any) {
      console.error("Error in analysis hook:", err);
      if (modelChoice === 'local') {
        setError(`Connection to the local model failed. Please ensure the Python backend server is running correctly. Check the "Model Integration Guide" for troubleshooting steps. (Details: ${err.message})`);
      } else {
        setError(err.message || 'An unknown error occurred during analysis.');
      }
      setIsLoading(false);
      return null;
    }
  }, []);

  return { analyze, isLoading, error };
};
