import { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleCompare = async () => {
    if(!file) {
      setError("Please select an image first.");
      return;
    }

    setError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/compare", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Backend error");
      }

      const data = await res.json();
      setResults(data.outputs);
    } catch (err) {
      setError("Failed to connect to backend.");
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: 40, fontFamily: "Arial" }}>
      <h1>Super Resolution Comparison Engine</h1>

      <input 
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])} 
      />
      <br /><br />

      <button onClick={handleCompare}>Compare</button>

      {loading && <p>Processing image...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {results && (
        <div style={{ display: "flex", gap: 20, marginTop: 30}}>
          {Object.entries(results).map(([name, path]) => (
            <div key={name}>
              <h3>{name.toUpperCase()}</h3>
              <img
                src={`http://127.0.0.1:8000/static/${path}`}
                width="300"
                alt={name}
              />
            </div>
          ))} 
        </div>
      )}
    </div>
  )
}

export default App;