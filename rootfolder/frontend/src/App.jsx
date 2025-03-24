// import { useState } from "react";

// function App() {
//   const [query, setQuery] = useState("");
//   const [image, setImage] = useState("");
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState("");
//   const [chartInfo, setChartInfo] = useState(null);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!query.trim()) {
//       setError("Please enter a query");
//       return;
//     }
//     setLoading(true);
//     setError("");
//     setImage("");
//     setChartInfo(null);

//     try {
//       const response = await fetch("http://localhost:8000/plot", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
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
//     <div className="container mx-auto p-4">
//       <h1 className="text-2xl font-bold mb-4">Data Visualization Tool</h1>
//       <form onSubmit={handleSubmit} className="mb-6">
//         <div className="mb-4">
//           <label htmlFor="query" className="block mb-2">
//             Enter your visualization query:
//           </label>
//           <textarea
//             id="w3review"
//             name="w3review"
//             rows="4"
//             cols="50"
//             type="textarea"
//             value={query}
//             onChange={(e) => setQuery(e.target.value)}
//             placeholder="e.g., Create a bar chart of monthly sales data"
//             className="w-full p-2 border rounded"
//           />
//         </div>
//         <button
//           type="submit"
//           disabled={loading}
//           className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-blue-300"
//         >
//           {loading ? "Generating..." : "Generate Visualization"}
//         </button>
//       </form>

//       {error && (
//         <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
//           {error}
//         </div>
//       )}
//       {loading && (
//         <div className="text-center py-8">Loading visualization...</div>
//       )}

//       {/* Display the visualization */}
//       {image && (
//         <div className="border rounded p-4 mt-4">
//           <h2 className="text-xl font-semibold mb-3">
//             {chartInfo?.title || "Generated Visualization"}
//           </h2>
//           <div className="mb-4 flex justify-center">
//             <img
//               style={{ height: "500px", width: "800px" }}
//               src={`data:image/png;base64,${image}`}
//               alt="Generated visualization"
//               className="max-w-full h-auto"
//             />
//           </div>
//           {chartInfo && (
//             <div className="bg-gray-100 p-3 rounded">
//               <p>
//                 <strong>Chart type:</strong> {chartInfo.chart_type}
//               </p>
//               <p>
//                 <strong>X-axis:</strong> {chartInfo.x_label}
//               </p>
//               <p>
//                 <strong>Y-axis:</strong> {chartInfo.y_label}
//               </p>
//             </div>
//           )}
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;

// For Chatbot purpose :
import { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const fetchSessions = async () => {
    try {
      const response = await fetch("http://localhost:8000/sessions");
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setSessions(data.sessions);
      
      //activate 1st session if no active session
      if (data.sessions.length > 0 && !activeSessionId) {
        setActiveSessionId(data.sessions[0].id);
        fetchSessionMessages(data.sessions[0].id);
      }
    } catch (err) {
      setError(`Failed to fetch sessions: ${err.message}`);
    }
  };

  //new chat sess created
  const createNewSession = async () => {
    try {
      setLoading(true);
      const response = await fetch("http://localhost:8000/sessions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: "New Chat" }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      await fetchSessions();
      setActiveSessionId(data.session_id);
      setMessages([]);
    } catch (err) {
      setError(`Failed to create new session: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionMessages = async (sessionId) => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/sessions/${sessionId}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      const data = await response.json();
      setMessages(data.messages || []);
    } catch (err) {
      setError(`Failed to fetch messages: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const switchSession = (sessionId) => {
    setActiveSessionId(sessionId);
    fetchSessionMessages(sessionId);
    setError("");
  };

  // Delete a chat session
  const deleteSession = async (sessionId, e) => {
    e.stopPropagation(); // Prevent activating the session when clicking delete
    
    try {
      const response = await fetch(`http://localhost:8000/sessions/${sessionId}`, {
        method: "DELETE",
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      // If we're deleting the active session, clear messages
      if (sessionId === activeSessionId) {
        setActiveSessionId(null);
        setMessages([]);
      }
      
      // Refresh sessions list
      await fetchSessions();
    } catch (err) {
      setError(`Failed to delete session: ${err.message}`);
    }
  };

  //sends msg
  const sendMessage = async (e) => {
    e.preventDefault();
    if (!newMessage.trim()) return;
    
    try {
      setLoading(true);
      setError("");
      
      //displaying the msg on screen
      const userMessage = { role: "user", content: newMessage };
      setMessages((prevMessages) => [...prevMessages, userMessage]);
      setNewMessage("");
      
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: activeSessionId,
          message: newMessage,
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      
      //update active session ID
      if (data.session_id !== activeSessionId) {
        setActiveSessionId(data.session_id);
        await fetchSessions();
      }
      
      // Add assistant response to messages
      if (data.response) {
        setMessages((prevMessages) => [...prevMessages, data.response]);
      }
      
    } catch (err) {
      setError(`Failed to send message: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <h1 className="app-title">Data Viz Chat</h1>
        </div>
        
        <div className="new-chat-container">
          <button
            onClick={createNewSession}
            className="new-chat-button"
            disabled={loading}
          >
            New Chat
          </button>
        </div>
        
        <div className="sessions-list">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => switchSession(session.id)}
              className={`session-item ${
                session.id === activeSessionId ? "active-session" : ""
              }`}
            >
              <div className="session-content">
                <div className="session-info">
                  <span className="session-name">Chat {session.id.substring(0, 6)}</span>
                  <p className="message-count">{session.message_count} messages</p>
                </div>
                <button
                  onClick={(e) => deleteSession(session.id, e)}
                  className="delete-button"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" height="16" width="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="chat-area">
        <div 
          ref={chatContainerRef}
          className="messages-container"
        >
          {activeSessionId ? (
            messages.length > 0 ? (
              <div className="messages-list">
                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`message-wrapper ${
                      msg.role === "user" ? "user-message-wrapper" : "assistant-message-wrapper"
                    }`}
                  >
                    <div
                      className={`message-bubble ${
                        msg.role === "user" ? "user-message" : "assistant-message"
                      }`}
                    >
                      <div className="message-content">{msg.content}</div>
                      {msg.image && (
                        <div className="message-image-container">
                          <img
                            src={`data:image/png;base64,${msg.image}`}
                            alt="Visualization"
                            className="message-image"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            ) : (
              <div className="empty-chat">
                <div className="empty-chat-message">
                  <h2 className="empty-chat-title">Data Visualization Assistant</h2>
                  <p>Ask me to create a chart or visualization for you!</p>
                  <p className="example-prompt">Example: "Create a line chart showing monthly sales data for 2023"</p>
                </div>
              </div>
            )
          ) : (
            <div className="empty-chat">
              <div className="empty-chat-message">
                <p>Select a chat or create a new one to start</p>
              </div>
            </div>
          )}
        </div>
        
        {error && (
          <div className="error-container">
            <p>{error}</p>
          </div>
        )}
        
        {activeSessionId && (
          <div className="input-container">
            <form onSubmit={sendMessage} className="message-form">
              <input
                type="text"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                placeholder="Enter your message or visualization request..."
                className="message-input"
                disabled={loading}
              />
              <button
                type="submit"
                className="send-button"
                disabled={loading || !newMessage.trim()}
              >
                {loading ? (
                  <svg className="loading-spinner" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">
                    <circle className="spinner-circle" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                    <path className="spinner-path" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  "Send"
                )}
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;