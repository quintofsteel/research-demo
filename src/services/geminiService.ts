
import { GoogleGenAI, Type, Modality } from '@google/genai';
import { ClinicalData, AnalysisResult } from '../types';

const API_KEY = process.env.API_KEY;
if (!API_KEY) {
  throw new Error("API_KEY environment variable not set. Please configure it to use the Gemini Simulation.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY, vertexai: true });

const fileToGenerativePart = async (file: File) => {
  const base64EncodedData = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = (err) => reject(err);
    reader.readAsDataURL(file);
  });

  return {
    inlineData: {
      data: base64EncodedData,
      mimeType: file.type,
    },
  };
};

const analysisSchema = {
  type: Type.OBJECT,
  properties: {
    prediction: {
      type: Type.STRING,
      enum: ["Pneumonia", "Normal"],
      description: "The final diagnosis prediction."
    },
    confidence: {
      type: Type.NUMBER,
      description: "The confidence score for the prediction, from 0.0 to 1.0."
    },
    shapAnalysis: {
      type: Type.STRING,
      description: "A detailed text explanation of feature importance, mimicking a SHAP analysis summary. Explain how age, sex, and view position influenced the prediction. If a value is not provided, state that."
    },
    fairnessAnalysis: {
        type: Type.STRING,
        description: "A brief analysis of potential fairness and bias considerations for this prediction, based on the provided demographic data and general knowledge from the thesis."
    }
  },
  required: ["prediction", "confidence", "shapAnalysis", "fairnessAnalysis"]
};

export const getPneumoniaAnalysis = async (
  imageFile: File,
  clinicalData: ClinicalData
): Promise<AnalysisResult> => {
  const imagePart = await fileToGenerativePart(imageFile);
  
  const clinicalDataText = `Clinical Data: Age: ${clinicalData.age || 'Not Provided'}, Sex: ${clinicalData.sex}, View Position: ${clinicalData.view_position}.`;

  const textPrompt = `You are an expert radiologist AI system based on the research paper 'Application of an Interpretable Machine Learning Model for Pneumonia Detection'. Your task is to analyze the provided chest X-ray and clinical data to detect pneumonia.
  
  ${clinicalDataText}
  
  Based on the image and data, provide a diagnosis. Also, generate a textual summary that explains how the clinical metadata influenced your decision (like SHAP) and a separate brief analysis on potential fairness considerations. Respond ONLY with the JSON object matching the provided schema.`;

  const gradCamPrompt = `Given this chest X-ray, simulate a Grad-CAM heatmap. Overlay a transparent, glowing heatmap in red and yellow on the areas most indicative of pneumonia. The heatmap should highlight pathological findings without obscuring the underlying anatomical structures. This is a simulation for a medical AI demo.`;

  const [analysisResponse, gradCamResponse] = await Promise.all([
    ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: {
        role: 'user',
        parts: [imagePart, { text: textPrompt }]
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: analysisSchema,
      }
    }),
    ai.models.generateContent({
      model: 'gemini-2.5-flash-image-preview',
      contents: {
        role: 'user',
        parts: [imagePart, { text: gradCamPrompt }]
      },
      config: {
        responseModalities: [Modality.IMAGE, Modality.TEXT],
      }
    })
  ]);

  let analysisData;
  try {
    analysisData = JSON.parse(analysisResponse.text);
  } catch (e) {
    console.error("Failed to parse analysis JSON:", analysisResponse.text);
    throw new Error("The model returned an invalid analysis format.");
  }

  const gradCamPart = gradCamResponse.candidates?.[0]?.content?.parts.find(part => part.inlineData);
  if (!gradCamPart || !gradCamPart.inlineData) {
    throw new Error("The model failed to generate a Grad-CAM image.");
  }
  const gradCamImage = `data:${gradCamPart.inlineData.mimeType};base64,${gradCamPart.inlineData.data}`;

  return {
    prediction: analysisData.prediction,
    confidence: analysisData.confidence,
    shapAnalysis: analysisData.shapAnalysis,
    gradCamImage: gradCamImage,
    fairnessAnalysis: analysisData.fairnessAnalysis,
  };
};
