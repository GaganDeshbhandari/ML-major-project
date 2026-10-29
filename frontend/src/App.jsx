import React from 'react';
import { ChevronRight, FileSpreadsheet, Gauge, LayoutDashboard, Menu } from 'lucide-react';
import { useRef, useState } from 'react';
import { navigation, objectives } from './data/sampleData';
import { createPrioritizationRun, previewDataset, validateDataset } from './api/client';
import DashboardPage from './pages/DashboardPage';
import RankedClaimsPage from './pages/RankedClaimsPage';
import ClaimDetailPage from './pages/ClaimDetailPage';
import ModelEvaluationPage from './pages/ModelEvaluationPage';

const icons = { dashboard: LayoutDashboard, claims: FileSpreadsheet, evaluation: Gauge };

export default function App() {
  const [activePage, setActivePage] = useState('Dashboard');
  const [claimsFile, setClaimsFile] = useState(null);
  const [perceptionFile, setPerceptionFile] = useState(null);
  const [claimsValidation, setClaimsValidation] = useState(null);
  const [perceptionValidation, setPerceptionValidation] = useState(null);
  const [claimsPreview, setClaimsPreview] = useState(null);
  const [perceptionPreview, setPerceptionPreview] = useState(null);
  const [objective, setObjective] = useState('false-claims');
  const [runMessage, setRunMessage] = useState('');
  const [isPreparingRun, setIsPreparingRun] = useState(false);
  const [runContext, setRunContext] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const claimsToken = useRef(0);
  const perceptionToken = useRef(0);

  const handleNavigation = (label) => {
    setActivePage(label);
    setMobileNavOpen(false);
  };

  const validateFile = async (file, datasetType, tokenRef, setValidation, setPreview) => {
    const token = tokenRef.current + 1;
    tokenRef.current = token;
    if (!file) {
      setValidation(null);
      setPreview(null);
      return;
    }
    setValidation({ status: 'checking' });
    setPreview(null);
    try {
      const [validation, preview] = await Promise.all([
        validateDataset(file, datasetType),
        previewDataset(file, datasetType),
      ]);
      if (token !== tokenRef.current) return;
      setValidation({ status: validation.valid ? 'valid' : 'invalid', validation });
      setPreview(preview);
    } catch (error) {
      if (token !== tokenRef.current) return;
      setValidation({ status: 'error', message: error.message || 'CSV validation failed.' });
    }
  };

  const handleClaimsFile = (file) => {
    setClaimsFile(file);
    setRunContext(null);
    setSelectedClaim(null);
    setRunMessage('');
    void validateFile(file, 'claims', claimsToken, setClaimsValidation, setClaimsPreview);
  };

  const handlePerceptionFile = (file) => {
    setPerceptionFile(file);
    setRunContext(null);
    setSelectedClaim(null);
    setRunMessage('');
    void validateFile(file, 'perception', perceptionToken, setPerceptionValidation, setPerceptionPreview);
  };

  const handleRun = async () => {
    const claimsReady = claimsValidation?.status === 'valid';
    const perceptionReady = perceptionValidation?.status === 'valid';
    if (!claimsFile || !perceptionFile || !claimsReady || !perceptionReady) {
      setRunMessage('Add both valid CSV files before starting a run.');
      return;
    }
    const selectedObjective = objectives.find((item) => item.id === objective);
    setIsPreparingRun(true);
    setRunMessage('');
    try {
      const run = await createPrioritizationRun({
        claimsFile,
        perceptionFile,
        objective: selectedObjective.apiValue,
      });
      setRunContext({
        runId: run.run_id,
        objective,
        apiObjective: run.objective,
        claimsFilename: claimsFile.name,
        perceptionFilename: perceptionFile.name,
        claimCount: run.claim_count,
        ratingCount: run.rating_count,
      });
      setSelectedClaim(null);
      setActivePage('Claims');
    } catch (error) {
      setRunMessage(error.message || 'Unable to start the prioritization run.');
    } finally {
      setIsPreparingRun(false);
    }
  };

  const startNewRun = () => {
    setRunContext(null);
    setSelectedClaim(null);
    setActivePage('Dashboard');
    setRunMessage('');
  };

  const page = activePage === 'Model evaluation' ? <ModelEvaluationPage />
    : activePage === 'Claims' ? <RankedClaimsPage runContext={runContext} onOpenClaim={(claim) => { setSelectedClaim(claim); setActivePage('ClaimDetail'); }} onStartNewRun={startNewRun} />
      : activePage === 'ClaimDetail' && selectedClaim ? <ClaimDetailPage claim={selectedClaim} runContext={runContext} onBack={() => setActivePage('Claims')} onStartNewRun={startNewRun} />
        : <DashboardPage {...{ claimsFile, perceptionFile, objective, setClaimsFile: handleClaimsFile, setPerceptionFile: handlePerceptionFile, setObjective, runMessage, setRunMessage, isPreparingRun, handleRun, claimsValidation, perceptionValidation, claimsPreview, perceptionPreview }} />;

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
      </aside>
      <main className="workspace">{page}</main>
    </div>
  );
}
