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
            <p>Ask me to visualize data for you. For example:</p>
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