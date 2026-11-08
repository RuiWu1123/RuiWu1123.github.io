import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';

const linkClass =
  'underline underline-offset-2 decoration-anthropic-text/30 hover:text-anthropic-text hover:decoration-anthropic-text transition-colors';

/**
 * The home page carries its own name in the introduction, so no header is drawn
 * there. Sub-pages get a link back home, and blog posts also one to the post list,
 * since there is no page menu.
 */
const Navbar: React.FC = () => {
  const { pathname } = useLocation();
  if (pathname === '/') return null;
  const inPost = pathname.startsWith('/thoughts/');

  return (
    <nav className="w-full">
      <div className="max-w-[680px] mx-auto px-6 pt-6 text-[13px] text-anthropic-gray">
        <NavLink to="/" className={linkClass}>
          Kris (Rui) Wu
        </NavLink>
        {inPost && (
          <>
            <span className="mx-2 text-anthropic-gray/50">/</span>
            <NavLink to="/thoughts" className={linkClass}>
              Thoughts &amp; Blog Posts
            </NavLink>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
