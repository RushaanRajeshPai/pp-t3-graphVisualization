// import React, { useState } from 'react';
// import './App.css';

// function App() {
//   const [query, setQuery] = useState('');
//   const [chatHistory, setChatHistory] = useState([]);
//   const [response, setResponse] = useState('');
//   const [graphImage, setGraphImage] = useState(null);
//   const [isLoading, setIsLoading] = useState(false);
//   const [error, setError] = useState(null);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!query.trim()) return;

//     setIsLoading(true);
//     setError(null);

//     try {
//       const response = await fetch('http://localhost:8000/chat', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({
//           query: query,
//           chat_history: chatHistory
//         })
//       });

//       if (!response.ok) {
//         throw new Error('Network response was not ok');
//       }

//       const data = await response.json();

//       // Update chat history
//       const newChatHistory = [
//         ...chatHistory, 
//         { type: 'user', text: query },
//         { type: 'assistant', text: data.response }
//       ];
//       setChatHistory(newChatHistory);

//       // Set response and graph
//       setResponse(data.response);
//       setGraphImage(data.graph ? `data:image/png;base64,${data.graph}` : null);
      
//       // Clear input
//       setQuery('');
//     } catch (error) {
//       console.error("Error:", error);
//       setError(error.message);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   return (
//     <div className="chat-container">
//       <div className="chat-header">
//         <h1>Data Visualization Chatbot</h1>
//       </div>
      
//       <div className="chat-messages">
//         {chatHistory.map((item, index) => (
//           <div 
//             key={index} 
//             className={`message ${item.type}-message`}
//           >
//             {item.text}
//           </div>
//         ))}
        
//         {graphImage && (
//           <div className="graph-container">
//             <img 
//               src={graphImage} 
//               alt="Generated Visualization" 
//               className="generated-graph"
//             />
//           </div>
//         )}
        
//         {isLoading && (
//           <div className="loading-indicator">
//             Generating response...
//           </div>
//         )}
        
//         {error && (
//           <div className="error-message">
//             {error}
//           </div>
//         )}
//       </div>
      
//       <form onSubmit={handleSubmit} className="chat-input-area">
//         <input 
//           type="text" 
//           value={query} 
//           onChange={(e) => setQuery(e.target.value)}
//           placeholder="Enter your data visualization query..."
//           disabled={isLoading}
//         />
//         <button 
//           type="submit" 
//           disabled={isLoading || !query.trim()}
//         >
//           {isLoading ? 'Processing...' : 'Generate'}
//         </button>
//       </form>
//     </div>
//   );
// }

// export default App;


// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// function App() {
//   const [query, setQuery] = useState('');
//   const [chatHistory, setChatHistory] = useState([]);
//   const [loading, setLoading] = useState(false);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
    
//     if (!query.trim()) return;

//     const userMessage = { role: 'user', content: query };
//     const updatedChatHistory = [...chatHistory, userMessage];
//     setChatHistory(updatedChatHistory);

//     setLoading(true);

//     try {
//       const response = await axios.post('http://localhost:8000/chat', {
//         query: query,
//         chat_history: updatedChatHistory
//       });

//       const botMessage = { 
//         role: 'bot', 
//         content: response.data.response,
//         graph: response.data.graph_base64 ? `data:image/png;base64,${response.data.graph_base64}` : null
//       };

//       setChatHistory(prev => [...prev, botMessage]);
//       setQuery('');
//     } catch (error) {
//       console.error('Error sending message:', error);
//       const errorMessage = { 
//         role: 'bot', 
//         content: 'Sorry, there was an error processing your request.' 
//       };
//       setChatHistory(prev => [...prev, errorMessage]);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="chat-container">
//       <div className="chat-messages">
//         {chatHistory.map((msg, index) => (
//           <div 
//             key={index} 
//             className={`message ${msg.role === 'user' ? 'user-message' : 'bot-message'}`}
//           >
//             {msg.content}
//             {msg.graph && (
//               <img 
//                 src={msg.graph} 
//                 alt="Generated Graph" 
//                 className="generated-graph"
//               />
//             )}
//           </div>
//         ))}
//         {loading && <div className="loading">Generating response...</div>}
//       </div>
//       <form onSubmit={handleSubmit} className="chat-input-form">
//         <input
//           type="text"
//           value={query}
//           onChange={(e) => setQuery(e.target.value)}
//           placeholder="Ask a question to generate a graph..."
//           className="chat-input"
//         />
//         <button type="submit" className="send-button" disabled={loading}>
//           Send
//         </button>
//       </form>
//     </div>
//   );
// }

// export default App;

// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// function App() {
//   const [query, setQuery] = useState('');
//   const [chatHistory, setChatHistory] = useState([]);
//   const [loading, setLoading] = useState(false);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!query.trim()) return;

//     const userMessage = { role: 'user', content: query };
//     const updatedHistory = [...chatHistory, userMessage];
//     setChatHistory(updatedHistory);
//     setLoading(true);

//     try {
//       const response = await axios.post('http://localhost:8000/chat', {
//         query,
//         chat_history: updatedHistory
//       });

//       const botMessage = {
//         role: 'bot',
//         content: response.data.response,
//         graph_base64: response.data.graph_base64 || null
//       };

//       setChatHistory(prev => [...prev, botMessage]);
//       setQuery('');
//     } catch (error) {
//       console.error('Error:', error);
//       setChatHistory(prev => [...prev, { role: 'bot', content: 'Server error.' }]);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="chat-container">
//       <div className="chat-box">
//         {chatHistory.map((msg, idx) => (
//           <div key={idx} className={`message ${msg.role}`}>
//             <p>{msg.content}</p>
//             {msg.graph_base64 && (
//               <img src={`data:image/png;base64,${msg.graph_base64}`} alt="Generated Graph" />
//             )}
//           </div>
//         ))}
//         {loading && <div className="message bot">Generating response...</div>}
//       </div>
//       <form onSubmit={handleSubmit} className="input-form">
//         <input
//           type="text"
//           value={query}
//           onChange={e => setQuery(e.target.value)}
//           placeholder="Enter your query..."
//         />
//         <button type="submit" disabled={loading}>Send</button>
//       </form>
//     </div>
//   );
// }

// export default App;

// File: src/App.jsx

import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      role: 'user',
      content: input,
    };

    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/plot', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages, // Send all messages for context
          query: input
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        image: data.image,
      };

      setMessages((prevMessages) => [...prevMessages, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error while processing your request.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <h1>Data Visualization Chatbot</h1>
      </header>
      
      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h2>Welcome to Data Visualization Chatbot!</h2>
            <p>Ask me to visualize data for you. For example:</p>
            <ul>
              <li>"Show me a graph of global temperature changes over the past 50 years"</li>
              <li>"Create a bar chart of the top 10 economies by GDP"</li>
              <li>"Visualize the population growth in major cities"</li>
            </ul>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`message ${message.role}`}>
              <div className="message-content">
                {message.content}
              </div>
              {message.image && (
                <div className="message-image">
                  <img 
                    src={`data:image/png;base64,${message.image}`} 
                    alt="Generated visualization" 
                  />
                </div>
              )}
            </div>
          ))
        )}
        {isLoading && (
          <div className="message assistant">
            <div className="loading-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me to visualize data..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Processing...' : 'Send'}
        </button>
      </form>
    </div>
  );
}

export default App;