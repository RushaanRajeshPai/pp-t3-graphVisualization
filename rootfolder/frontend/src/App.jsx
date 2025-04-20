// src/App.js
import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [socket, setSocket] = useState(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    let ws = null;
  
    const connectWebSocket = () => {
      ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        console.log('WebSocket Connected');
        setIsConnected(true);
        setSocket(ws);
      };
  
      ws.onclose = () => {
        console.log('WebSocket Disconnected');
        setIsConnected(false);
        setSocket(null);
      };
  
      ws.onerror = (error) => {
        console.error('WebSocket Error:', error);
      };
  
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received:', data); // Debug log
          
          if (data.status === 'completed') {
            handleResponse(data.result);
          } else if (data.status === 'error') {
            setMessages(prev => [...prev, {
              type: 'error',
              content: data.message
            }]);
          }
        } catch (error) {
          console.error('Parse error:', error);
        }
      };
    };
  
    connectWebSocket();
  
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  useEffect(() => {
    // Auto-scroll to bottom when messages update
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) return;

    // Add user query to messages
    setMessages(prevMessages => [
      ...prevMessages,
      {
        type: 'user',
        content: query
      }
    ]);

    setIsProcessing(true);

    // If WebSocket is connected, use it
    if (isConnected && socket) {
      // try {
      //   socket.send(JSON.stringify({ query }));
      // } catch (error) {
      //   console.error('WebSocket error:', error);
      //   // Fallback to REST API
      //   try {
      //     const response = await fetch('http://localhost:8000/api/query', {
      //       method: 'POST',
      //       headers: {
      //         'Content-Type': 'application/json',
      //       },
      //       body: JSON.stringify({ query }),
      //     });
      //     const data = await response.json();
      //     handleResponse(data);
      //   } catch (error) {
      //     console.error('API error:', error);
      //     setMessages(prevMessages => [
      //       ...prevMessages,
      //       {
      //         type: 'error',
      //         content: 'Failed to process query. Please try again.'
      //       }
      //     ]);
      //   }
      // }

      try {
        const response = await fetch('http://localhost:8000/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        const data = await response.json();
        handleResponse(data);
      } catch (error) {
        console.error('API error:', error);
      }
      
    } else {
      // REST API fallback
      try {
        const response = await fetch('http://localhost:8000/api/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ query }),
        });
        const data = await response.json();
        handleResponse(data);
      } catch (error) {
        console.error('API error:', error);
        setMessages(prevMessages => [
          ...prevMessages,
          {
            type: 'error',
            content: 'Failed to process query. Please try again.'
          }
        ]);
      }
    }

    setQuery('');
    setIsProcessing(false);
  };

  // Add a helper function to handle responses
  const handleResponse = (data) => {
    setMessages(prevMessages => [
      ...prevMessages,
      {
        type: 'assistant',
        content: data.response,
        hasGraph: data.has_graph,
        graphData: data.graph_data
      }
    ]);
  };

  // Function to render markdown content
  const renderMarkdown = (content) => {
    // You'll need to install a package like react-markdown
    // This is a simplified version for demonstration
    return content;
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold text-blue-700">Research Paper Assistant</h1>
        <p className="text-white-600 mt-2 ">Ask questions about the research paper to get accurate responses</p>
      </header>

      <div
        ref={chatContainerRef}
        className="bg-white rounded-lg shadow-md p-6 mb-6 h-[60vh] overflow-y-auto"
      >
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <p>Ask a question about the research paper to begin</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`mb-4 ${message.type === 'user' ? 'text-right' : ''}`}>
              <div className={`inline-block p-4 rounded-lg ${message.type === 'user'
                  ? 'bg-blue-100 text-blue-800'
                  : message.type === 'error'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-gray-100 text-gray-800'
                }`}>
                <div className="prose">
                  {renderMarkdown(message.content)}
                </div>

                {message.hasGraph && message.graphData && message.graphData.image_base64 && (
                  <div className="mt-4">
                    <img
                      src={`data:image/png;base64,${message.graphData.image_base64}`}
                      alt="Generated graph"
                      className="max-w-full rounded-lg shadow-sm"
                    />
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {isProcessing && (
          <div className="flex items-center space-x-2 text-gray-600 mt-2">
            <span>{processingStatus || 'Processing'}</span>
            <div className="loading-dots flex space-x-1">
              <div className="w-2 h-2 bg-gray-600 rounded-full"></div>
              <div className="w-2 h-2 bg-gray-600 rounded-full"></div>
              <div className="w-2 h-2 bg-gray-600 rounded-full"></div>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="flex items-center space-x-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about the research paper..."
          className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isProcessing}
        />
        <button
          type="submit"
          className={`bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-lg ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          disabled={isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Ask'}
        </button>
      </form>

      <div className="mt-6 text-center text-sm text-gray-500">
        <p>Connection status: {isConnected ? 'Connected' : 'Disconnected'}</p>
      </div>
    </div>
  );
}

export default App;