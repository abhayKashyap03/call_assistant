import { useState } from 'react';
import './App.css';

// Import the two "page" components we have created
import MainPage from './MainPage';
import SettingsPage from './SettingsPage';
import DocIn from './DocIn';

// Define the possible pages our app can show
type Page = 'main' | 'settings' | 'docin';

export default function App() {
  // --- STATE MANAGEMENT ---
  // This state variable keeps track of which page is currently active.
  const [currentPage, setCurrentPage] = useState<Page>('main');

  // --- UI RENDERING (JSX) ---
  return (
    <div className="container">
      <header>
        <h1>AI Call Agent</h1>
        {/* The navigation changes which component is rendered below */}
        <nav className="main-nav">
          <button 
            onClick={() => setCurrentPage('main')}
            className={currentPage === 'main' ? 'active' : ''}>
            Control Panel
          </button>
          <button 
            onClick={() => setCurrentPage('settings')}
            className={currentPage === 'settings' ? 'active' : ''}>
            Settings
          </button>
          <button
            onClick={() => setCurrentPage('docin')}
            className={currentPage === 'docin' ? 'active' : ''}>
            Add Documents
          </button>
        </nav>
      </header>

      <main>
        {/* --- CONDITIONAL RENDERING --- */}
        {/* This is the logic that switches between pages */}
        {currentPage === 'main' && <MainPage />}
        {currentPage === 'settings' && <SettingsPage onSaveSuccess={() => setCurrentPage('main')}/>}
        {currentPage === 'docin' && <DocIn />}
      </main>
    </div>
  );
}