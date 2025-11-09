
import { ClinicalData, AnalysisResult } from '../types';

const API_URL = 'http://localhost:5000/api/analyze';

export const getLocalModelAnalysis = async (
  imageFile: File,
  clinicalData: ClinicalData
): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('image', imageFile);
  // Append clinical data only if it exists
  if (clinicalData.age) formData.append('age', clinicalData.age);
  if (clinicalData.sex) formData.append('sex', clinicalData.sex);
  if (clinicalData.view_position) formData.append('view_position', clinicalData.view_position);

  const response = await fetch(API_URL, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    try {
        const errorData = JSON.parse(errorText);
        throw new Error(errorData.error || `Server responded with status ${response.status}`);
    } catch {
        throw new Error(errorText || `Server responded with status ${response.status}`);
    }
  }

  const result = await response.json();
  
  // The backend returns base64 strings for images. The frontend needs data URLs.
  // The result from the backend should match the AnalysisResult structure,
  // but gradCamImage will be a raw base64 string.
  return {
      ...result,
      gradCamImage: `data:image/png;base64,${result.gradCamImage}`
  };
};
