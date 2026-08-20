import React, { useState, useEffect, useRef } from 'react';
import type { Message } from '../App';

interface ChatPaneProps {
  messages: Message[];
  onSendMessage: (text: string) => void;
  streamData: any;
  onSimulateDemo?: () => void;
  onClearData?: () => void;
}

const ChatPane: React.FC<ChatPaneProps> = ({ 
  messages, 
  onSendMessage, 
  streamData,
  onSimulateDemo,
  onClearData 
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [highlightMessageId, setHighlightMessageId] = useState<string | null>(null);
  const [highlightColor, setHighlightColor] = useState<string>('');

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (streamData && streamData.nodes && streamData.nodes.length > 0) {
      // Find the last user message to highlight
      const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
      if (lastUserMsg) {
        // Just pick the category of the first node for the highlight color
        const firstNode = streamData.nodes[0];
        let cat = 'entity';
        if (firstNode.label === 'Fact') cat = 'topic';
        else if (firstNode.label === 'Turn') cat = 'event';
        else if (firstNode.label === 'Alias') cat = 'preference';

        setHighlightMessageId(lastUserMsg.id);
        setHighlightColor(`var(--cat-${cat}-glow)`);

        // Remove highlight after animation
        setTimeout(() => setHighlightMessageId(null), 1500);
      }
    }
  }, [streamData]); // messages dependency omitted to only trigger on new streamData

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSendMessage(input.trim());
    setInput('');
  };

  return (
    <>
      <div className="chat-header">
        <span className="title pixel-font">HYDRADB CHAT</span>
        <div className="chat-header-actions">
          {onSimulateDemo && (
            <button 
              type="button" 
              className="header-btn pixel-font simulate-btn" 
              onClick={onSimulateDemo}
            >
              SIMULATE DEMO DATA
            </button>
          )}
          {onClearData && (
            <button 
              type="button" 
              className="header-btn pixel-font clear-btn" 
              onClick={onClearData}
            >
              CLEAR
            </button>
          )}
        </div>
      </div>
      <div className="chat-messages">
        {messages.map(msg => (
          <div 
            key={msg.id} 
            className={`message ${msg.role} ${highlightMessageId === msg.id ? 'highlight-pulse' : ''}`}
            style={highlightMessageId === msg.id ? { '--glow-color': highlightColor } as React.CSSProperties : {}}
          >
            <div className="message-meta pixel-font">
              {msg.role.toUpperCase()} • {new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
            </div>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-input-container" onSubmit={handleSubmit}>
        <input 
          className="chat-input pixel-font"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message to ingest..."
        />
        <button type="submit" className="send-btn pixel-font">SEND</button>
      </form>
    </>
  );
};

export default ChatPane;
