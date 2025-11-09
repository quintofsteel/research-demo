
import React from 'react';
import { LogoIcon } from './icons';

export const Header: React.FC = () => {
  return (
    <header className="bg-slate-800 shadow-lg">
      <div className="container mx-auto px-4 py-4 flex items-center space-x-4">
        <LogoIcon className="h-10 w-10 text-sky-400" />
        <div>
          <h1 className="text-xl font-bold text-white">
            Interpretable AI for Pneumonia Detection
          </h1>
          <p className="text-sm text-slate-300">
            A Demonstration of Multimodal Fusion, Generalization, and Fairness
          </p>
        </div>
      </div>
    </header>
  );
};
