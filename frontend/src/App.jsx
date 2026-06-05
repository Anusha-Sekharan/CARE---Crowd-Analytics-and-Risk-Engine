import { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Connect to FastAPI running on port 8000
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getAlertColor = (level) => {
    switch (level) {
      case "GREEN": return "bg-green-100 text-green-800 border-green-400";
      case "YELLOW": return "bg-yellow-100 text-yellow-800 border-yellow-400";
      case "ORANGE": return "bg-orange-100 text-orange-800 border-orange-400";
      case "RED": return "bg-red-100 text-red-800 border-red-400";
      default: return "bg-gray-100 text-gray-800 border-gray-400";
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-900">Crowd Guardian</h1>
          <p className="text-gray-600 mt-2">Multi-Agent Crowd Risk Detection & Early Warning System</p>
        </header>

        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-8">
          <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-12 hover:border-blue-500 transition-colors">
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleFileChange} 
              className="mb-4 text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            {preview && (
              <img src={preview} alt="Preview" className="max-h-64 rounded shadow-sm mb-4" />
            )}
            <button 
              onClick={handleUpload}
              disabled={!file || loading}
              className={`px-6 py-2 rounded-full font-medium text-white transition-all ${
                !file || loading ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700 shadow-md hover:shadow-lg"
              }`}
            >
              {loading ? "Analyzing Workflow..." : "Analyze Crowd"}
            </button>
          </div>
          {error && (
            <div className="mt-4 p-4 bg-red-50 text-red-700 rounded border border-red-200">
              Error: {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        {results && (
          <div className="space-y-6">
            
            {/* Alert Banner */}
            <div className={`p-4 rounded-lg border-l-4 shadow-sm flex items-start space-x-4 ${getAlertColor(results.alert.alert_level)}`}>
              <div className="flex-1">
                <h3 className="font-bold text-lg mb-1">Alert Level: {results.alert.alert_level}</h3>
                <p>{results.alert.warning_message}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Telemetry Panel */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-xl font-bold mb-4 border-b pb-2">Telemetry</h2>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Count:</span>
                    <span className="font-bold">{results.telemetry.people_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Density Score:</span>
                    <span className="font-bold">{results.telemetry.density_score}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Hotspot Zone:</span>
                    <span className="font-bold">{results.telemetry.hotspot_zone}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Zone Max Count:</span>
                    <span className="font-bold">{results.telemetry.max_zone_count}</span>
                  </div>
                </div>
              </div>

              {/* Risk Panel */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-xl font-bold mb-4 border-b pb-2">Risk Prediction</h2>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Current Risk:</span>
                    <span className="font-bold">{results.risk.risk_level}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Congestion Score:</span>
                    <span className="font-bold">{results.risk.congestion_score}/100</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Future Est. Count:</span>
                    <span className="font-bold">{results.risk.future_people_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 text-red-600 font-medium">Stampede Risk:</span>
                    <span className="font-bold text-red-600">{results.risk.stampede_risk}%</span>
                  </div>
                </div>
              </div>

              {/* LLM Advisor Panel */}
              <div className="bg-white rounded-xl shadow-md p-6">
                <h2 className="text-xl font-bold mb-4 border-b pb-2">LLM Advisor</h2>
                <div className="space-y-3 text-sm">
                  <div>
                    <h4 className="font-semibold text-gray-800">Summary</h4>
                    <p className="text-gray-600">{results.advisor.summary}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-800">Recommendations</h4>
                    <p className="text-gray-600 whitespace-pre-line">{results.advisor.recommendations}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Images Visualizer */}
            <div className="bg-white rounded-xl shadow-md p-6">
              <h2 className="text-xl font-bold mb-4 border-b pb-2">Visual Analysis</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium text-gray-700 mb-2 text-center">Original</h4>
                  {results.images.original && (
                    <img src={`http://localhost:8000${results.images.original}`} alt="Original" className="rounded-lg w-full" />
                  )}
                </div>
                <div>
                  <h4 className="font-medium text-gray-700 mb-2 text-center">Heatmap Overlay</h4>
                  {results.images.heatmap ? (
                    <img src={`http://localhost:8000${results.images.heatmap}`} alt="Heatmap" className="rounded-lg w-full" />
                  ) : (
                    <div className="flex items-center justify-center h-full bg-gray-100 rounded-lg text-gray-400">
                      No Heatmap Generated
                    </div>
                  )}
                </div>
              </div>
            </div>

          </div>
        )}
      </div>
    </div>
  );
}

export default App;
