import Link from 'next/link';
import CodeExecution from '../components/CodeExecution';

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
          
          <div className="prose max-w-none mb-8">
            <p className="text-gray-600">
              Our machine learning model analyzes clinical symptoms to predict the most likely genetic neurological disorders.
              Enter the patient's symptoms in the code editor below to get predictions.
            </p>
          </div>

          <CodeExecution />
        </div>
      </main>
    </div>
  );
} 