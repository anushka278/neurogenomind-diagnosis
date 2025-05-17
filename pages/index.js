import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-gray-800">NeuroGenomind Diagnosis</h1>
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

      <main className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-8">
            Bridging the Diagnostic Odyssey
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Development of Machine Learning-Based Predictive Models for Genetic Neurological Disorder Diagnosis
          </p>
          <div className="mt-8">
            <Link href="/background" 
                  className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
              Learn More
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
} 