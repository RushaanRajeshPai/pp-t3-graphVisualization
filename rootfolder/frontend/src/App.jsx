// import { useState } from 'react';

// function App() {
//   const [query, setQuery] = useState('');
//   const [image, setImage] = useState('');
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [chartInfo, setChartInfo] = useState(null);

//   const handleSubmit = async (e) => {
//     e.preventDefault();

//     if (!query.trim()) {
//       setError('Please enter a query');
//       return;
//     }

//     setLoading(true);
//     setError('');
//     setImage('');
//     setChartInfo(null);

//     try {
//       const response = await fetch('http://localhost:8000/plot', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ query }),
//       });

//       if (!response.ok) {
//         throw new Error(`Error: ${response.status}`);
//       }

//       const data = await response.json();
//       setImage(data.image);
//       setChartInfo(data.config);
//     } catch (err) {
//       setError(`Failed to generate plot: ${err.message}`);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div>
//       <h1>Data Visualization Tool</h1>

//       <form onSubmit={handleSubmit}>
//         <div>
//           <label htmlFor="query">Enter your visualization query:</label>
//           <input
//             type="text"
//             id="query"
//             value={query}
//             onChange={(e) => setQuery(e.target.value)}
//             placeholder="Enter query"
//           />
//         </div>

//         <button type="submit" disabled={loading}>
//           {loading ? 'Generating...' : 'Generate Visualization'}
//         </button>
//       </form>

//       {error && <div>{error}</div>}

//       {loading && <div>Loading visualization...</div>}

//       {chartInfo && (
//         <div>
//           <h2>{chartInfo.title}</h2>
//           {image && (
//             <div>
//               <img 
//                 src={`data:image/png;base64,${image}`} 
//                 alt={chartInfo.title} 
//               />
//               {/* <pre style={{wordWrap:"break-word", whiteSpace:"pre-wrap"}}>
//                 {image}
//               </pre> */}
//               <div>
//                 <p>Chart type: {chartInfo.chart_type}</p>
//                 <p>X-axis: {chartInfo.x_label}</p>
//                 <p>Y-axis: {chartInfo.y_label}</p>
//               </div>
//             </div>
//           )}
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;

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
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Data Visualization Tool</h1>
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="mb-4">
          <label htmlFor="query" className="block mb-2">Enter your visualization query:</label>
          <input
            type="text"
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Create a bar chart of monthly sales data"
            className="w-full p-2 border rounded"
          />
        </div>
        <button 
          type="submit" 
          disabled={loading}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-blue-300"
        >
          {loading ? 'Generating...' : 'Generate Visualization'}
        </button>
      </form>

      {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">{error}</div>}
      {loading && <div className="text-center py-8">Loading visualization...</div>}

      {chartInfo && (
        <div className="border rounded p-4">
          <h2 className="text-xl font-semibold mb-3">{chartInfo.title}</h2>
          {image && (
            <div>
              <div className="mb-4 flex justify-center">
                <img 
                  src={`data:image/png;base64,${image}`} 
                  alt={chartInfo.title} 
                  className="max-w-full"
                />
              </div>
              <div className="bg-gray-100 p-3 rounded">
                <p><strong>Chart type:</strong> {chartInfo.chart_type}</p>
                <p><strong>X-axis:</strong> {chartInfo.x_label}</p>
                <p><strong>Y-axis:</strong> {chartInfo.y_label}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;