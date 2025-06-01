import { useState } from 'react';
import Editor from '@monaco-editor/react';
import axios from 'axios';

export default function CodeExecution() {
  const [code, setCode] = useState(`# Enter symptoms here
symptoms = [
    "seizures",
    "developmental delay",
    "microcephaly"
]

# The model will analyze these symptoms and return
# the top 20 most likely genetic neurological disorders`);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleEditorChange = (value) => {
    setCode(value);
  };

  const executeCode = async () => {
    setLoading(true);
    setError(null);
    try {
      // Extract symptoms from the code
      const symptomsMatch = code.match(/symptoms\s*=\s*\[([\s\S]*?)\]/);
      if (!symptomsMatch) {
        throw new Error('Please define symptoms as a list in the code');
      }

      const symptomsStr = symptomsMatch[1];
      const symptoms = symptomsStr
        .split(',')
        .map(s => s.trim().replace(/['"]/g, ''))
        .filter(s => s);

      // TODO: Replace with actual API endpoint
      const response = await axios.post('/api/predict', { symptoms });
      setResults(response.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="p-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Input Symptoms</h3>
          <p className="mt-1 text-sm text-gray-500">
            Enter the patient's symptoms in the code editor below
          </p>
        </div>
        <div className="h-[300px]">
          <Editor
            height="100%"
            defaultLanguage="python"
            theme="vs-dark"
            value={code}
            onChange={handleEditorChange}
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              lineNumbers: 'on',
              roundedSelection: false,
              scrollBeyondLastLine: false,
              automaticLayout: true,
            }}
          />
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={executeCode}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Running...' : 'Run Analysis'}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {results && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Analysis Results</h3>
          <div className="space-y-4">
            {results.predictions.map((prediction, index) => (
              <div key={index} className="border-b border-gray-200 pb-4 last:border-0">
                <div className="flex justify-between items-center">
                  <span className="font-medium text-gray-900">{prediction.disorder}</span>
                  <span className="text-sm text-gray-500">{prediction.probability}%</span>
                </div>
                <p className="mt-1 text-sm text-gray-600">{prediction.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 