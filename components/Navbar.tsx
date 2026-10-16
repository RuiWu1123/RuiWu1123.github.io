import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';

/**
 * The header carries the name only. Alignment and Blogs are reached from links
 * inside the home page's introduction, so there is no page menu; the name is
 * the way back from a sub-page.
 */
const Navbar: React.FC = () => {
  const { pathname } = useLocation();
  const atHome = pathname === '/';

  return (
    <nav className="w-full border-b border-anthropic-text/10">
      <div className="max-w-[680px] mx-auto px-6 h-16 flex items-center">
        <NavLink
          to="/"
          className="text-lg font-serif tracking-tight hover:opacity-70 transition-opacity"
          aria-current={atHome ? 'page' : undefined}
        >
          Kris Rui Wu
        </NavLink>
      </div>
    </nav>
  );
};

export default Navbar;
