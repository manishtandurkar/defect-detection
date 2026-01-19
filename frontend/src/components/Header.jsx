import React from 'react';
import { ScanLine, Sparkles } from 'lucide-react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <div className="logo-icon">
            <ScanLine size={36} strokeWidth={2.5} />
          </div>
          <div className="logo-text">
            <h1>Defect Detection System</h1>
            <div className="badge">
              <Sparkles size={14} />
              <span>XAI Powered</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
