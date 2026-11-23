import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';

const linkClass =
  'underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';

/**
 * The home page carries its own name in the introduction, so no header is drawn
 * there. Sub-pages get a link back home, since there is no page menu.
 */
const Navbar: React.FC = () => {
  const { pathname } = useLocation();
  if (pathname === '/') return null;

  return (
    <nav className="w-full">
      <div className="max-w-[680px] mx-auto px-6 pt-6 text-[13px] text-anthropic-gray">
        <NavLink to="/" className={linkClass}>
          Kris (Rui) Wu
        </NavLink>
      </div>
    </nav>
  );
};

export default Navbar;
