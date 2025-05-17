import Link from 'next/link';

export default function Problem() {
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
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Problem Statement</h1>
          
          <div className="prose max-w-none">
            <h2 className="text-2xl font-semibold mb-4">Current Challenges</h2>
            <p className="mb-4">
              Keeping updated on the vast number of possible genetic neurological syndromes, many of which are rare disorders, presents a significant challenge for healthcare providers. The current Online Mendelian Inheritance in Man (OMIM) database lists thousands of genetic syndromes that have manifestations of neurological symptoms.
            </p>

            <p className="mb-4">
              Commonly used searching systems for professionals, such as PubMed, are usually highly sophisticated and time-consuming for searching through the contents. This creates a significant barrier to efficient diagnosis and treatment planning.
            </p>

            <h2 className="text-2xl font-semibold mb-4 mt-8">Proposed Solution</h2>
            <p className="mb-4">
              We propose the development of a machine learning-based assistive tool designed to accelerate the diagnostic process for suspected genetic neurological disorders. This tool will:
            </p>

            <ul className="list-disc pl-6 mb-4">
              <li>Take user-provided clinical symptoms (phenotypes) as input</li>
              <li>Leverage the knowledge curated in the OMIM database</li>
              <li>Output a list of the top 20 most likely genetic neurological disorder diagnoses</li>
              <li>Provide a user-friendly web interface for clinicians</li>
            </ul>

            <h2 className="text-2xl font-semibold mb-4 mt-8">Expected Impact</h2>
            <p className="mb-4">
              Such a tool could:
            </p>
            <ul className="list-disc pl-6 mb-4">
              <li>Facilitate earlier diagnosis and intervention</li>
              <li>Help avoid unnecessary diagnostic testing</li>
              <li>Improve patient's quality of life</li>
              <li>Reduce the time and effort required for diagnosis</li>
            </ul>
          </div>
        </div>
      </main>
    </div>
  );
} 