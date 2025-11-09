
import React from 'react';

interface ToggleProps<T extends string> {
  label: string;
  option1: { value: T; label: string };
  option2: { value: T; label: string };
  value: T;
  onChange: (value: T) => void;
}

export function Toggle<T extends string>({ label, option1, option2, value, onChange }: ToggleProps<T>) {
  const isOption1 = value === option1.value;

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-2">{label}</label>
      <div className="relative flex w-full rounded-lg bg-slate-200 p-1">
        <span
          className="absolute top-1 bottom-1 left-1 w-1/2 rounded-md bg-white shadow-md transition-transform duration-300 ease-in-out"
          style={{ transform: isOption1 ? 'translateX(0%)' : 'translateX(100%)' }}
        />
        <button
          type="button"
          onClick={() => onChange(option1.value)}
          className={`relative z-10 w-1/2 py-2 text-sm font-medium rounded-md transition-colors ${isOption1 ? 'text-slate-800' : 'text-slate-500 hover:text-slate-700'}`}
          aria-pressed={isOption1}
        >
          {option1.label}
        </button>
        <button
          type="button"
          onClick={() => onChange(option2.value)}
          className={`relative z-10 w-1/2 py-2 text-sm font-medium rounded-md transition-colors ${!isOption1 ? 'text-slate-800' : 'text-slate-500 hover:text-slate-700'}`}
          aria-pressed={!isOption1}
        >
          {option2.label}
        </button>
      </div>
    </div>
  );
}
