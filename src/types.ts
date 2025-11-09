
export type ModelChoice = 'gemini' | 'local';

export interface ClinicalData {
  age: string;
  sex: 'Male' | 'Female';
  view_position: 'AP' | 'PA';
}

export interface AnalysisResult {
  prediction: 'Pneumonia' | 'Normal';
  confidence: number;
  shapAnalysis: string;
  gradCamImage: string;
  fairnessAnalysis: string;
}
