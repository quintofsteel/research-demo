
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { XRayIcon } from './icons';

interface ImageUploaderProps {
  file: File | null;
  setFile: (file: File | null) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ file, setFile }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, [setFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/dicom': ['.dcm'] },
    maxFiles: 1,
  });

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
  };

  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1">
        Chest X-Ray Image
      </label>
      <div
        {...getRootProps()}
        className={`mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-slate-300 border-dashed rounded-lg cursor-pointer transition-colors duration-200 ease-in-out ${
          isDragActive ? 'bg-sky-50 border-sky-400' : 'bg-white hover:bg-slate-50'
        }`}
      >
        <input {...getInputProps()} />
        {file ? (
          <div className="text-center relative">
            <img
              src={URL.createObjectURL(file)}
              alt="X-ray preview"
              className="max-h-48 rounded-md mx-auto shadow-md"
            />
            <p className="text-sm text-slate-600 mt-2 font-medium">{file.name}</p>
            <button
              onClick={removeFile}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full h-6 w-6 flex items-center justify-center text-xs font-bold hover:bg-red-600 transition-transform transform hover:scale-110"
              aria-label="Remove image"
            >
              &times;
            </button>
          </div>
        ) : (
          <div className="space-y-1 text-center">
            <XRayIcon className="mx-auto h-12 w-12 text-slate-400" />
            <div className="flex text-sm text-slate-600">
              <p className="pl-1">
                {isDragActive
                  ? 'Drop the image here...'
                  : 'Drag & drop an image, or click to select'}
              </p>
            </div>
            <p className="text-xs text-slate-500">PNG, JPG, or DICOM up to 10MB</p>
          </div>
        )}
      </div>
    </div>
  );
};
