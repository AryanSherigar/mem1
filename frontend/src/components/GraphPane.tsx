import React, { useEffect, useRef, useState } from 'react';

interface GraphPaneProps {
  streamData: any;
  onClose: () => void;
}

const categories: Record<string, { color: string, glow: string }> = {
  entity:     { color: '#7F77DD', glow: 'rgba(127,119,221,0.55)' },
  topic:      { color: '#1D9E75', glow: 'rgba(29,158,117,0.55)' },
  event:      { color: '#D85A30', glow: 'rgba(216,90,48,0.55)' },
  preference: { color: '#D4537E', glow: 'rgba(212,83,126,0.55)' }
};

interface Node {
  id: string;
  label: string;
  cat: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  targetR: number;
  born: number;
  alpha: number;
}

interface Edge {
  a: Node;
  b: Node;
  born: number;
  alpha: number;
}

const GraphPane: React.FC<GraphPaneProps> = ({ streamData, onClose }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const edgesRef = useRef<Edge[]>([]);
  const requestRef = useRef<number | undefined>(undefined);
  
  const [nodeCount, setNodeCount] = useState(0);
  const [detailCard, setDetailCard] = useState<{node: Node, x: number, y: number} | null>(null);

  // Viewport transforms
  const transform = useRef({ x: 0, y: 0, scale: 1 });
  const isDragging = useRef(false);
  const dragNode = useRef<Node | null>(null);
  const lastMouse = useRef({ x: 0, y: 0 });

  // Handle incoming data
  useEffect(() => {
    if (!streamData || !canvasRef.current) return;
    
    if (streamData.type === 'graph_clear') {
      nodesRef.current = [];
      edgesRef.current = [];
      setNodeCount(0);
      setDetailCard(null);
      return;
    }

    if (streamData.type === 'graph_update') {
      const W = canvasRef.current.width / (window.devicePixelRatio || 1);
      const H = canvasRef.current.height / (window.devicePixelRatio || 1);

      // Add Nodes
      streamData.nodes.forEach((nData: any) => {
        if (nodesRef.current.some(n => n.id === nData.id)) return;

        let labelText = nData.properties.name || nData.properties.content || nData.label;
        if (labelText && labelText.length > 20) labelText = labelText.substring(0, 18) + '..';
        
        let cat = 'entity';
        if (nData.label === 'Fact') cat = 'topic';
        else if (nData.label === 'Turn') cat = 'event';
        else if (nData.label === 'Alias') cat = 'preference';

        const angle = Math.random() * Math.PI * 2;
        const dist = 40 + Math.random() * 20;
        
        nodesRef.current.push({
          id: nData.id,
          label: labelText.toUpperCase(),
          cat,
          x: W/2 + Math.cos(angle)*dist,
          y: H/2 + Math.sin(angle)*dist,
          vx: 0, vy: 0,
          r: 0, targetR: 7 + Math.random() * 3,
          born: performance.now(),
          alpha: 0
        });
      });

      setNodeCount(nodesRef.current.length);

      // Add Edges
      setTimeout(() => {
        streamData.edges.forEach((eData: any) => {
          const a = nodesRef.current.find(n => n.id === eData.source_id);
          const b = nodesRef.current.find(n => n.id === eData.target_id);
          if (!a || !b) return;

          const exists = edgesRef.current.some(e => (e.a === a && e.b === b) || (e.a === b && e.b === a));
          if (!exists) {
            edgesRef.current.push({ a, b, born: performance.now(), alpha: 0 });
          }
        });
      }, 300);
    }
  }, [streamData]);

  // Main Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const rect = canvas.parentElement!.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener('resize', resize);

    const step = () => {
      const now = performance.now();
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.width / dpr;
      const H = canvas.height / dpr;
      const cx = W / 2, cy = H / 2;

      // Physics
      for (const n of nodesRef.current) {
        if (n === dragNode.current) continue; // Skip physics for dragged node

        const age = now - n.born;
        n.alpha = Math.min(1, age / 400);
        const spring = 1 - Math.pow(1 - Math.min(1, age / 500), 3);
        n.r = n.targetR * spring;

        let fx = (cx - n.x) * 0.0009;
        let fy = (cy - n.y) * 0.0009;

        for (const o of nodesRef.current) {
          if (o === n) continue;
          const dx = n.x - o.x, dy = n.y - o.y;
          const d = Math.sqrt(dx * dx + dy * dy + 0.01);
          if (d < 90) {
            const f = (90 - d) / d * 0.06;
            fx += dx * f; fy += dy * f;
          }
        }
        n.vx = (n.vx + fx) * 0.9;
        n.vy = (n.vy + fy) * 0.9;
        n.x += n.vx;
        n.y += n.vy;
      }

      for (const e of edgesRef.current) {
        const dx = e.b.x - e.a.x, dy = e.b.y - e.a.y;
        const d = Math.sqrt(dx*dx + dy*dy) || 1;
        const f = (d - 90) * 0.002;
        const fx = (dx / d) * f, fy = (dy / d) * f;
        if (e.a !== dragNode.current) { e.a.vx += fx; e.a.vy += fy; }
        if (e.b !== dragNode.current) { e.b.vx -= fx; e.b.vy -= fy; }
        e.alpha = Math.min(1, (now - e.born) / 500);
      }

      // Draw
      ctx.clearRect(0, 0, W, H);
      
      ctx.save();
      ctx.translate(transform.current.x, transform.current.y);
      ctx.scale(transform.current.scale, transform.current.scale);

      // Draw Edges
      for (const e of edgesRef.current) {
        const grad = ctx.createLinearGradient(e.a.x, e.a.y, e.b.x, e.b.y);
        grad.addColorStop(0, categories[e.a.cat]?.color || '#fff');
        grad.addColorStop(1, categories[e.b.cat]?.color || '#fff');
        
        ctx.strokeStyle = grad;
        ctx.globalAlpha = e.alpha * 0.35;
        ctx.lineWidth = 1.5;
        
        const mx = (e.a.x + e.b.x) / 2 + (e.a.y - e.b.y) * 0.06;
        const my = (e.a.y + e.b.y) / 2 + (e.b.x - e.a.x) * 0.06;
        
        ctx.beginPath();
        ctx.moveTo(e.a.x, e.a.y);
        ctx.quadraticCurveTo(mx, my, e.b.x, e.b.y);
        ctx.stroke();
      }

      ctx.globalAlpha = 1;

      // Draw Nodes
      for (const n of nodesRef.current) {
        const c = categories[n.cat] || categories.entity;
        ctx.save();
        ctx.globalAlpha = n.alpha;
        ctx.shadowColor = c.glow;
        ctx.shadowBlur = 16;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = c.color;
        ctx.fill();
        ctx.restore();

        if (n.alpha > 0.6) {
          ctx.save();
          ctx.globalAlpha = (n.alpha - 0.6) / 0.4;
          ctx.font = '9px "Press Start 2P", monospace';
          ctx.fillStyle = '#c8c8d2';
          ctx.textAlign = 'center';
          ctx.fillText(n.label, n.x, n.y + n.r + 18);
          ctx.restore();
        }
      }

      ctx.restore();
      requestRef.current = requestAnimationFrame(step);
    };

    requestRef.current = requestAnimationFrame(step);

    return () => {
      window.removeEventListener('resize', resize);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // Interaction handlers
  const getMousePos = (e: React.MouseEvent | React.WheelEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  };

  const getTransformedPos = (x: number, y: number) => {
    return {
      x: (x - transform.current.x) / transform.current.scale,
      y: (y - transform.current.y) / transform.current.scale
    };
  };

  const onMouseDown = (e: React.MouseEvent) => {
    const pos = getMousePos(e);
    lastMouse.current = pos;
    const tPos = getTransformedPos(pos.x, pos.y);

    // Check if clicking a node
    for (let i = nodesRef.current.length - 1; i >= 0; i--) {
      const n = nodesRef.current[i];
      const dx = tPos.x - n.x;
      const dy = tPos.y - n.y;
      if (Math.sqrt(dx*dx + dy*dy) < n.r + 5) {
        dragNode.current = n;
        setDetailCard(null);
        return;
      }
    }
    
    // Otherwise pan
    isDragging.current = true;
    setDetailCard(null);
  };

  const onMouseMove = (e: React.MouseEvent) => {
    const pos = getMousePos(e);
    const dx = pos.x - lastMouse.current.x;
    const dy = pos.y - lastMouse.current.y;

    if (dragNode.current) {
      dragNode.current.x += dx / transform.current.scale;
      dragNode.current.y += dy / transform.current.scale;
    } else if (isDragging.current) {
      transform.current.x += dx;
      transform.current.y += dy;
    }
    
    lastMouse.current = pos;
  };

  const onMouseUp = () => {
    if (dragNode.current) {
      // If barely moved, show detail card
      const n = dragNode.current;
      setDetailCard({
        node: n,
        x: n.x * transform.current.scale + transform.current.x,
        y: n.y * transform.current.scale + transform.current.y
      });
    }
    isDragging.current = false;
    dragNode.current = null;
  };

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const pos = getMousePos(e);
    const zoomIntensity = 0.1;
    const wheel = e.deltaY < 0 ? 1 : -1;
    const scale = Math.exp(wheel * zoomIntensity);
    
    const newScale = transform.current.scale * scale;
    if (newScale < 0.1 || newScale > 10) return;

    transform.current.x = pos.x - (pos.x - transform.current.x) * scale;
    transform.current.y = pos.y - (pos.y - transform.current.y) * scale;
    transform.current.scale = newScale;
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div className="graph-header">
        <div className="graph-header-left">
          <div className="pulse-dot" style={{ animation: streamData ? 'flash 0.5s ease' : 'none' }} />
          <span className="graph-label pixel-font">HYDRADB GRAPH</span>
        </div>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center', pointerEvents: 'auto' }}>
          <span className="node-count pixel-font">{nodeCount} NODES</span>
          <button className="close-btn pixel-font" onClick={onClose}>X</button>
        </div>
      </div>
      
      <canvas 
        ref={canvasRef} 
        className="graph-canvas"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onWheel={onWheel}
      />

      {detailCard && (
        <div className="detail-card" style={{ left: detailCard.x, top: detailCard.y }}>
          <div className="detail-card-header">
            <span className="detail-cat" style={{ background: categories[detailCard.node.cat]?.color }}>
              {detailCard.node.cat.toUpperCase()}
            </span>
            <span className="detail-close" onClick={() => setDetailCard(null)}>×</span>
          </div>
          <div className="detail-content">{detailCard.node.label}</div>
          <div className="detail-time pixel-font">ID: {detailCard.node.id.split('-')[0]}...</div>
        </div>
      )}
    </div>
  );
};

export default GraphPane;
