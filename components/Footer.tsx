import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="w-full py-12 border-t border-anthropic-text/5 mt-auto">
      <div className="max-w-6xl mx-auto px-6 text-center text-anthropic-gray text-sm font-sans">
        <p>© {new Date().getFullYear()} Rui Wu. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;