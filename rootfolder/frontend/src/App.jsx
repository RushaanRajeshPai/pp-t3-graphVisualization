import { useState } from 'react';

function App() {
  const [query, setQuery] = useState('');
  const [image, setImage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [chartInfo, setChartInfo] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) {
      setError('Please enter a query');
      return;
    }

    setLoading(true);
    setError('');
    setImage('');
    setChartInfo(null);

    try {
      const response = await fetch('http://localhost:8000/plot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setImage(data.image);
      setChartInfo(data.config);
    } catch (err) {
      setError(`Failed to generate plot: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Data Visualization Tool</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="query">Enter your visualization query:</label>
          <input
            type="text"
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Number of job losses vs. number of years"
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Generating...' : 'Generate Visualization'}
        </button>
      </form>

      {error && <div>{error}</div>}

      {loading && <div>Loading visualization...</div>}

      {chartInfo && (
        <div>
          <h2>{chartInfo.title}</h2>
          {image && (
            <div>
              {/* <img 
                src={`data:image/png;base64,${image}`} 
                alt={chartInfo.title} 
              /> */}
              <pre style={{ wordWrap: "break-word", whiteSpace: "pre-wrap" }}>
                {image}
              </pre>
              <div>
                <p>Chart type: {chartInfo.chart_type}</p>
                <p>X-axis: {chartInfo.x_label}</p>
                <p>Y-axis: {chartInfo.y_label}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;