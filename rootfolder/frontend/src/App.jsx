// src/App.jsx
import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentState, setCurrentState] = useState('idle');
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
        // Try to reconnect in 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
  
      ws.onerror = (error) => {
        console.error('WebSocket Error:', error);
      };
  
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received:', data);
          
          if (data.status === 'processing') {
            setCurrentState(data.state || 'processing_query');
          } else if (data.status === 'completed') {
            handleResponse(data.result);
            setCurrentState('completed');
          } else if (data.status === 'error') {
            setMessages(prev => [...prev, {
              type: 'error',
              content: data.message
            }]);
            setCurrentState('error');
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
    setCurrentState('processing_query');

    // If WebSocket is connected, use it
    if (isConnected && socket) {
      try {
        socket.send(JSON.stringify({ query }));
      } catch (error) {
        console.error('WebSocket error:', error);
        // Fallback to REST API
        await fetchWithRestApi();
      }
    } else {
      // REST API fallback
      await fetchWithRestApi();
    }

    setQuery('');
  };

  const fetchWithRestApi = async () => {
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
      setCurrentState('completed');
    } catch (error) {
      console.error('API error:', error);
      setMessages(prevMessages => [
        ...prevMessages,
        {
          type: 'error',
          content: 'Failed to process query. Please try again.'
        }
      ]);
      setCurrentState('error');
    } finally {
      setIsProcessing(false);
    }
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
    setIsProcessing(false);
  };

  // Get status message based on current state
  const getStatusMessage = () => {
    const stateMessages = {
      'idle': 'Ready',
      'processing_query': 'Processing query',
      'enhancing_query': 'Enhancing query',
      'establishing_context': 'Establishing context',
      'retrieving_data': 'Retrieving data from research paper',
      'merging_response': 'Merging information',
      'detecting_contradictions': 'Detecting contradictions',
      'resolving_contradictions': 'Resolving contradictions',
      'labeling_sources': 'Labeling sources',
      'summarizing': 'Summarizing content',
      'planning_graph': 'Planning visualization',
      'generating_graph': 'Generating visualization',
      'generating_response': 'Generating final response',
      'storing_memory': 'Storing in memory',
      'completed': 'Completed',
      'error': 'Error occurred'
    };
    
    return stateMessages[currentState] || 'Processing';
  };

  // Function to render markdown content
  const renderMarkdown = (content) => {
    // This is a simplified version
    // For proper markdown rendering, install and use a package like react-markdown
    return (
      <div dangerouslySetInnerHTML={{ __html: content.replace(/\n/g, '<br />') }} />
    );
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <header className="text-center mb-8">
        <h1 className="text-3xl font-bold text-blue-700">Research Paper Assistant</h1>
        <p className="text-gray-600 mt-2">Ask questions about research papers</p>
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
              <div className={`inline-block p-4 rounded-lg max-w-[80%] ${
                message.type === 'user'
                  ? 'bg-blue-100 text-blue-800'
                  : message.type === 'error'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-gray-100 text-gray-800'
              }`}>
                <div className="prose max-w-none">
                  {renderMarkdown(message.content)}
                </div>

                {message.hasGraph && message.graphData && message.graphData.image_base64 && (
                  <div className="mt-4">
                    <img
                      src={`data:image/png;base64,${message.graphData.image_base64}`}
                      alt={`${message.graphData.graph_type || 'Generated'} graph`}
                      className="max-w-full rounded-lg shadow-sm"
                    />
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {isProcessing && (
          <div className="flex items-center space-x-2 text-gray-600 mt-4 ml-4">
            <span>{getStatusMessage()}</span>
            <div className="loading-dots flex space-x-1">
              <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              <div className="w-2 h-2 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
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
          className={`bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-lg transition ${
            isProcessing ? 'opacity-50 cursor-not-allowed' : ''
          }`}
          disabled={isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Ask'}
        </button>
      </form>

      <div className="mt-6 flex justify-between text-sm text-gray-500">
        <p>Connection status: <span className={isConnected ? "text-green-500" : "text-red-500"}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span></p>
        <p>Current state: <span className="font-medium">{getStatusMessage()}</span></p>
      </div>
    </div>
  );
}

export default App;