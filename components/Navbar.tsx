import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { NAV_ITEMS } from '../constants';

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 w-full bg-anthropic-bg border-b border-anthropic-text/10">
      <div className="max-w-[760px] mx-auto px-6 h-20 flex items-center justify-between">
        {/* Logo / Name */}
        <NavLink to="/" className="text-2xl font-serif font-medium tracking-tight hover:opacity-70 transition-opacity">
          Rui Wu
        </NavLink>

        {/* Desktop Menu */}
        <div className="hidden md:flex space-x-8">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `text-sm font-sans tracking-wide transition-colors duration-200 ${
                  isActive ? 'text-anthropic-text underline underline-offset-[6px] decoration-anthropic-text/40' : 'text-anthropic-gray hover:text-anthropic-text'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden p-2 text-anthropic-text"
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle menu"
        >
          {isOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Menu Dropdown */}
      {isOpen && (
        <div className="md:hidden bg-anthropic-bg border-b border-anthropic-text/5 px-6 py-4 space-y-4 shadow-sm">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `block text-base font-sans ${
                  isActive ? 'text-anthropic-text underline underline-offset-4 decoration-anthropic-text/40' : 'text-anthropic-gray'
                }`
              }
              onClick={() => setIsOpen(false)}
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      )}
    </nav>
  );
};

export default Navbar;