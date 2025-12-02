import React, { Component, ErrorInfo, ReactNode } from 'react';
import { HashRouter, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Blog from './pages/Blog';
import Publication from './pages/Publication';
import Travel from './pages/Travel';

// Scroll to top on route change
const ScrollToTop = () => {
  const { pathname } = useLocation();
  React.useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
};

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col font-sans text-anthropic-text bg-anthropic-bg selection:bg-anthropic-accent/20">
      <Navbar />
      <main className="flex-grow w-full max-w-6xl mx-auto px-6">
        {children}
      </main>
      <Footer />
    </div>
  );
};

// Error Boundary to catch runtime crashes
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean, error: Error | null }> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-[#F4F3EF] text-[#191919] p-8">
          <div className="max-w-md">
            <h1 className="text-2xl font-serif mb-4">Something went wrong</h1>
            <p className="font-sans mb-4">The application encountered an unexpected error.</p>
            <pre className="bg-gray-200 p-4 rounded text-xs overflow-auto mb-4">
              {this.state.error?.message}
            </pre>
            <button 
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-[#D97757] text-white rounded hover:opacity-90 transition-opacity"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <HashRouter>
        <ScrollToTop />
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/blog" element={<Blog />} />
            <Route path="/publications" element={<Publication />} />
            <Route path="/travel" element={<Travel />} />
          </Routes>
        </Layout>
      </HashRouter>
    </ErrorBoundary>
  );
};

export default App;