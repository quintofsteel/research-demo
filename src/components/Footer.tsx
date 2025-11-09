
import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-200 mt-12">
      <div className="container mx-auto py-4 px-4 text-center text-slate-600 text-sm">
        <p>
          Inspired by the thesis: "Application of an Interpretable Machine Learning Model for Pneumonia Detection in Chest X-Rays and Clinical Data" (HRCOS82).
        </p>
        <p>This is a demonstration application. Not for clinical use.</p>
      </div>
    </footer>
  );
};
