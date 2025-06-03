import Link from "next/link";
import { useState } from "react";

export default function Home() {
  const [symptoms, setSymptoms] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePrediction = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          symptoms: symptoms.split(",").map((s) => s.trim()),
        }),
      });
      const data = await response.json();
      setPredictions(data.predictions);
    } catch (error) {
      console.error("Error getting predictions:", error);
    }
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <nav className="bg-white/80 backdrop-blur-md shadow-lg border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  NeuroGenomind Diagnosis
                </h1>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <Link
                href="/background"
                className="text-gray-600 hover:text-blue-600 transition-colors font-medium"
              >
                Background
              </Link>
              <Link
                href="/problem"
                className="text-gray-600 hover:text-blue-600 transition-colors font-medium"
              >
                Problem
              </Link>
              <Link
                href="/model"
                className="text-gray-600 hover:text-blue-600 transition-colors font-medium"
              >
                Model
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
            Neurological Diagnosis Assistant
          </h1>
          <p className="text-xl text-gray-600">
            Enter patient symptoms below to receive AI-powered diagnostic
            predictions
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100">
          <div className="space-y-6">
            <div>
              <label
                htmlFor="symptoms"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Patient Symptoms
              </label>
              <textarea
                id="symptoms"
                rows="4"
                className="w-full px-4 py-3 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
                placeholder="Enter symptoms separated by commas (e.g., headache, dizziness, fatigue)"
                value={symptoms}
                onChange={(e) => setSymptoms(e.target.value)}
              />
            </div>

            <button
              onClick={handlePrediction}
              disabled={isLoading || !symptoms.trim()}
              className={`w-full py-3 px-6 rounded-lg text-white font-medium transition-all transform hover:scale-[1.02] ${
                isLoading || !symptoms.trim()
                  ? "bg-gray-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
              }`}
            >
              {isLoading ? "Analyzing..." : "Run Prediction"}
            </button>
          </div>
        </div>

        {predictions && (
          <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Predicted Diagnoses
            </h2>
            <div className="space-y-4">
              {predictions.map((prediction, index) => (
                <div
                  key={index}
                  className="p-4 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-100"
                >
                  <div className="flex justify-between items-center">
                    <span className="font-medium text-gray-800">
                      {prediction.condition}
                    </span>
                    <span className="text-sm font-medium text-blue-600">
                      {(prediction.probability * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
