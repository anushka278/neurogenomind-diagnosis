import Link from 'next/link';

export default function Background() {
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
          <h1 className="text-3xl font-bold text-gray-900 mb-6">Background</h1>
          
          <div className="prose max-w-none">
            <p className="mb-4">
              Neurological disorders have emerged as the leading cause of combined disability and mortality globally over the past three decades. A recent systematic analysis estimated that more than 3 billion people worldwide were living with a neurological condition in 2021.
            </p>

            <p className="mb-4">
              The World Health Organization (WHO) has made neurological disorders a significant priority in global health policy, which is reflected in the implementation of the Intersectoral Global Action Plan on Epilepsy and other Neurological Disorders 2022-31 (IGAP) and the initiation of the WHO Brain Health Initiative.
            </p>

            <p className="mb-4">
              According to the WHO, the top ten neurological conditions contributing to health loss in 2021 included:
            </p>

            <ul className="list-disc pl-6 mb-4">
              <li>Stroke</li>
              <li>Neonatal encephalopathy</li>
              <li>Epilepsy</li>
              <li>Migraine</li>
              <li>Dementia</li>
              <li>Diabetic neuropathy</li>
              <li>Meningitis</li>
              <li>Neurological complications from preterm birth</li>
              <li>Autism spectrum disorder (ASD)</li>
              <li>Nervous system cancers</li>
            </ul>

            <p className="mb-4">
              Many of these neurological diseases, such as ASD, epilepsy, and neonatal encephalopathy, are complex diseases known to have a strong genetic component. The current Online Mendelian Inheritance in Man (OMIM) database, one of the largest reference databases of human genetic disorders, lists thousands of genetic syndromes that had manifestations of neurological symptoms.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
} 