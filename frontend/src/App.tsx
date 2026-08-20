import React, { useState, useEffect, useRef } from 'react';
import ChatPane from './components/ChatPane';
import GraphPane from './components/GraphPane';
import './styles/theme.css';

export interface Message {
  id: string;
  role: 'user' | 'agent';
  content: string;
  timestamp: string;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [graphPaneOpen, setGraphPaneOpen] = useState(true);
  const [graphWidth, setGraphWidth] = useState(60); // percentage
  const [isResizing, setIsResizing] = useState(false);
  const appRef = useRef<HTMLDivElement>(null);
  const [streamData, setStreamData] = useState<any>(null); // For SSE events

  const contextId = "demo-context";
  const sessionId = "demo-session";

  // Handle resizing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !appRef.current) return;
      const rect = appRef.current.getBoundingClientRect();
      const newWidth = ((rect.right - e.clientX) / rect.width) * 100;
      if (newWidth > 20 && newWidth < 80) {
        setGraphWidth(newWidth);
      }
    };

    const handleMouseUp = () => setIsResizing(false);

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Setup SSE
  useEffect(() => {
    const streamUrl = window.location.port === '5173' 
      ? 'http://localhost:8000/v1/memory/stream' 
      : '/v1/memory/stream';

    const eventSource = new EventSource(streamUrl);

    eventSource.onmessage = (event) => {
      if (event.data === 'keepalive') return;
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'graph_update' || payload.type === 'graph_clear') {
          setStreamData(payload);
        } else if (payload.type === 'chat_message') {
          setMessages(prev => {
            if (prev.some(m => m.id === payload.message.id)) return prev;
            return [...prev, payload.message];
          });
        }
      } catch (err) {
        console.error("Failed to parse SSE payload", err);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE EventSource failed", err);
      // It auto-reconnects
    };

    return () => eventSource.close();
  }, []);

  const handleSimulateDemo = async () => {
    const simUrl = window.location.port === '5173' 
      ? 'http://localhost:8000/v1/demo/simulate' 
      : '/v1/demo/simulate';
    try {
      setMessages([]);
      setStreamData({ type: 'graph_clear' });
      await fetch(simUrl, { method: 'POST' });
    } catch (err) {
      console.error("Failed to trigger demo simulation", err);
    }
  };

  const handleClearData = async () => {
    const clearUrl = window.location.port === '5173' 
      ? 'http://localhost:8000/v1/demo/clear' 
      : '/v1/demo/clear';
    try {
      setMessages([]);
      setStreamData({ type: 'graph_clear' });
      await fetch(clearUrl, { method: 'POST' });
    } catch (err) {
      console.error("Failed to clear data", err);
    }
  };

  const handleSendMessage = async (text: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, newMessage]);

    try {
      const response = await fetch('http://localhost:8000/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          context_id: contextId,
          session_id: sessionId,
          user_message: text
        })
      });

      const data = await response.json();
      const replyMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'agent',
        content: data.reply,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, replyMessage]);
    } catch (err) {
      console.error("Chat error", err);
    }
  };

  return (
    <div className="app-container" ref={appRef}>
      <div className="pane chat-pane" style={{ width: graphPaneOpen ? `${100 - graphWidth}%` : '100%' }}>
        <ChatPane 
          messages={messages} 
          onSendMessage={handleSendMessage} 
          streamData={streamData} 
          onSimulateDemo={handleSimulateDemo}
          onClearData={handleClearData}
        />
        {!graphPaneOpen && (
          <button className="toggle-btn pixel-font" onClick={() => setGraphPaneOpen(true)}>
            SHOW GRAPH
          </button>
        )}
      </div>

      {graphPaneOpen && (
        <>
          <div 
            className={`resizer ${isResizing ? 'dragging' : ''}`} 
            onMouseDown={() => setIsResizing(true)} 
          />
          <div className="pane graph-pane" style={{ width: `${graphWidth}%` }}>
            <GraphPane 
              streamData={streamData} 
              onClose={() => setGraphPaneOpen(false)} 
            />
          </div>
        </>
      )}
    </div>
  );
};

export default App;
