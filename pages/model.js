import Link from 'next/link';

export default function Model() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <Link href="/" className="text-xl font-bold text-gray-800">NeuroGenomind Diagnosis</Link>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/background" className="text-gray-600 hover:text-gray-900">Background</Link>
              <Link href="/problem" className="text-gray-600 hover:text-gray-900">Problem</Link>
              <Link href="/model" className="text-gray-600 hover:text-gray-900">Model</Link>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="bg-white shadow-lg rounded-lg p-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Model Development</h1>
          
          <div className="prose max-w-none">
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-8">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-yellow-700">
                    Model development is currently in progress. Check back soon for updates!
                  </p>
                </div>
              </div>
            </div>

            <h2 className="text-2xl font-semibold mb-4">Planned Features</h2>
            <ul className="list-disc pl-6 mb-4">
              <li>Machine learning model trained on OMIM database</li>
              <li>Input processing for clinical symptoms</li>
              <li>Output of top 20 most likely diagnoses</li>
              <li>Integration with existing medical databases</li>
              <li>User-friendly interface for healthcare providers</li>
            </ul>

            <h2 className="text-2xl font-semibold mb-4 mt-8">Evaluation Metrics</h2>
            <p className="mb-4">
              The model will be evaluated using:
            </p>
            <ul className="list-disc pl-6 mb-4">
              <li>Top-20 Accuracy</li>
              <li>Mean Reciprocal Rank (MRR)</li>
              <li>Comparison with baseline LLM models</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
} 