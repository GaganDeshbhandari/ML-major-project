import { ChevronRight, CircleHelp, FileSpreadsheet, Gauge, LayoutDashboard, ListChecks, Menu, Settings } from 'lucide-react';
import { useState } from 'react';
import { navigation } from './data/sampleData';
import DashboardPage from './pages/DashboardPage';
import RankedClaimsPage from './pages/RankedClaimsPage';
import ClaimDetailPage from './pages/ClaimDetailPage';
import ModelEvaluationPage from './pages/ModelEvaluationPage';

const icons = { dashboard: LayoutDashboard, runs: ListChecks, claims: FileSpreadsheet, evaluation: Gauge, settings: Settings };

export default function App() {
  const [activePage, setActivePage] = useState('Dashboard');
  const [claimsFile, setClaimsFile] = useState(null);
  const [perceptionFile, setPerceptionFile] = useState(null);
  const [objective, setObjective] = useState('false-claims');
  const [runMessage, setRunMessage] = useState('');
  const [isPreparingRun, setIsPreparingRun] = useState(false);
  const [runContext, setRunContext] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const handleNavigation = (label) => {
    setActivePage(label); setMobileNavOpen(false);
    setRunMessage(label !== 'Dashboard' && label !== 'Claims' && label !== 'Model evaluation' ? `${label} is not part of this first dashboard prototype yet.` : '');
  };
  const handleRun = () => {
    if (!claimsFile || !perceptionFile) { setRunMessage('Add both valid CSV files before preparing the sample ranking.'); return; }
    setIsPreparingRun(true); setRunMessage('');
    window.setTimeout(() => {
      setRunContext({ objective, claimsFilename: claimsFile.name, perceptionFilename: perceptionFile.name });
      setSelectedClaim(null); setActivePage('Claims'); setIsPreparingRun(false);
    }, 650);
  };

  const page = activePage === 'Model evaluation' ? <ModelEvaluationPage />
    : activePage === 'Claims' ? <RankedClaimsPage runContext={runContext} onOpenClaim={(claim) => { setSelectedClaim(claim); setActivePage('ClaimDetail'); }} />
      : activePage === 'ClaimDetail' && selectedClaim ? <ClaimDetailPage claim={selectedClaim} runContext={runContext} onBack={() => setActivePage('Claims')} />
        : <DashboardPage {...{ claimsFile, perceptionFile, objective, setClaimsFile, setPerceptionFile, setObjective, runMessage, setRunMessage, isPreparingRun, handleRun }} />;

  return (
    <div className="app-shell">
      <button className="mobile-menu-button" aria-label="Open navigation" onClick={() => setMobileNavOpen((isOpen) => !isOpen)}><Menu size={21} /></button>
      <aside className={`sidebar ${mobileNavOpen ? 'sidebar--open' : ''}`}>
        <div className="brand"><div className="brand-mark">CR</div><div><div className="brand-name">ClaimRank</div><div className="brand-subtitle">AI</div></div></div>
        <div className="sidebar-label">Workspace</div>
        <nav className="sidebar-nav" aria-label="Main navigation">
          {navigation.map(({ label, iconName }) => {
            const Icon = icons[iconName];
            const isActive = activePage === label || (label === 'Claims' && activePage === 'ClaimDetail');
            return <button className={`nav-item ${isActive ? 'nav-item--active' : ''}`} key={label} onClick={() => handleNavigation(label)}><Icon size={18} strokeWidth={isActive ? 2.4 : 2} /><span>{label}</span>{activePage === label && <ChevronRight className="nav-arrow" size={16} />}</button>;
          })}
        </nav>
        <div className="sidebar-footer"><div className="help-icon"><CircleHelp size={17} /></div><div><strong>Need help?</strong><span>Read the documentation</span></div></div>
      </aside>
      <main className="workspace">{page}</main>
    </div>
  );
}
