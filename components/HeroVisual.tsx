import React, { useEffect, useRef } from 'react';

interface Leaf {
  x: number;
  y: number;
  size: number;
  speedY: number;
  speedX: number;
  angle: number;
  spin: number;
  color: string;
}

const HeroVisual: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let leaves: Leaf[] = [];
    
    // Resize handler
    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        // Ensure we don't set 0 dimensions which can cause issues
        canvas.width = Math.max(parent.clientWidth, 300);
        canvas.height = Math.max(parent.clientHeight, 300);
      } else {
        canvas.width = window.innerWidth;
        canvas.height = 400;
      }
      
      initLeaves();
    };

    const colors = ['#8DA399', '#D97757', '#6B6B6B', '#E6E4DD']; // Leaf, Accent, Gray, Stone

    const initLeaves = () => {
      leaves = [];
      const count = 30; // Number of leaves
      for (let i = 0; i < count; i++) {
        leaves.push(createLeaf(true));
      }
    };

    const createLeaf = (randomY = false): Leaf => {
      const size = Math.random() * 12 + 8;
      // Safety check for width
      const safeWidth = canvas.width || 800;
      const safeHeight = canvas.height || 400;
      
      return {
        x: Math.random() * safeWidth,
        y: randomY ? Math.random() * safeHeight : -50,
        size: size,
        speedY: Math.random() * 0.4 + 0.2,
        speedX: Math.random() * 0.4 - 0.2,
        angle: Math.random() * Math.PI * 2,
        spin: (Math.random() - 0.5) * 0.02,
        color: colors[Math.floor(Math.random() * colors.length)]
      };
    };

    const drawLeaf = (leaf: Leaf) => {
      if (!ctx) return;
      ctx.save();
      ctx.translate(leaf.x, leaf.y);
      ctx.rotate(leaf.angle);
      
      ctx.beginPath();
      // Draw a simple leaf shape
      ctx.ellipse(0, 0, leaf.size / 2, leaf.size, 0, 0, Math.PI * 2);
      
      ctx.fillStyle = leaf.color;
      ctx.globalAlpha = 0.5;
      ctx.fill();
      ctx.restore();
    };

    const animate = () => {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      leaves.forEach((leaf, index) => {
        // Update position
        leaf.y += leaf.speedY;
        leaf.x += Math.sin(leaf.y * 0.01) * 0.5 + leaf.speedX;
        leaf.angle += leaf.spin;

        // Reset if out of bounds
        if (leaf.y > canvas.height + 50) {
          leaves[index] = createLeaf();
        }

        drawLeaf(leaf);
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    window.addEventListener('resize', resize);
    
    // Initial setup
    resize();
    animate();

    return () => {
      window.removeEventListener('resize', resize);
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full"
      style={{ opacity: 0.8 }}
    />
  );
};

export default HeroVisual;