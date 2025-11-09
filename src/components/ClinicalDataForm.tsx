
import React from 'react';
import { ClinicalData } from '../types';

interface ClinicalDataFormProps {
  data: ClinicalData;
  setData: (data: ClinicalData) => void;
}

export const ClinicalDataForm: React.FC<ClinicalDataFormProps> = ({ data, setData }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setData({ ...data, [e.target.name]: e.target.value });
  };

  return (
    <div>
      <h3 className="text-sm font-medium text-slate-700 mb-2">Clinical Metadata (Optional)</h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div>
          <label htmlFor="age" className="block text-xs font-medium text-slate-600">
            Age
          </label>
          <input
            type="number"
            name="age"
            id="age"
            value={data.age}
            onChange={handleChange}
            className="mt-1 block w-full px-3 py-2 bg-white border border-slate-300 rounded-md shadow-sm focus:outline-none focus:ring-sky-500 focus:border-sky-500 sm:text-sm"
            placeholder="e.g., 55"
          />
        </div>
        <div>
          <label htmlFor="sex" className="block text-xs font-medium text-slate-600">
            Sex
          </label>
          <select
            id="sex"
            name="sex"
            value={data.sex}
            onChange={handleChange}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-slate-300 focus:outline-none focus:ring-sky-500 focus:border-sky-500 sm:text-sm rounded-md"
          >
            <option>Male</option>
            <option>Female</option>
          </select>
        </div>
        <div>
          <label htmlFor="view_position" className="block text-xs font-medium text-slate-600">
            View Position
          </label>
          <select
            id="view_position"
            name="view_position"
            value={data.view_position}
            onChange={handleChange}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-slate-300 focus:outline-none focus:ring-sky-500 focus:border-sky-500 sm:text-sm rounded-md"
          >
            <option>AP</option>
            <option>PA</option>
          </select>
        </div>
      </div>
    </div>
  );
};
