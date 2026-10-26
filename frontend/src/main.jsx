import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  BarChart3,
  CheckCircle2,
  Copy,
  Check,
  ChevronRight,
  CircleHelp,
  Download,
  ExternalLink,
  ArrowLeft,
  FileSpreadsheet,
  Gauge,
  Info,
  LayoutDashboard,
  ListChecks,
  Menu,
  Search,
  Settings,
  UploadCloud,
  X,
} from 'lucide-react';
import './styles.css';

const navigation = [
  { label: 'Dashboard', icon: LayoutDashboard },
  { label: 'Prioritization runs', icon: ListChecks },
  { label: 'Claims', icon: FileSpreadsheet },
  { label: 'Model evaluation', icon: Gauge },
  { label: 'Settings', icon: Settings },
];

const objectives = [
  {
    id: 'false-claims',
    title: 'Remove False Claims',
    description: 'Prioritize claims most likely to be false.',
  },
  {
    id: 'misconceptions',
    title: 'Correct User-Misconceptions',
    description: 'Prioritize claims that users perceive incorrectly.',
  },
  {
    id: 'disagreement',
    title: 'Reduce User Disagreement',
    description: 'Prioritize claims with the most disagreement.',
  },
];

const sampleClaims = [
  { id: 'CLM-0248', claim: "The city's new policy reduced traffic congestion by 40%.", baseScore: 0.91, ecs: 0.82, priority: 'High', status: 'Mixed evidence', source: 'City transport report and local traffic data show different baselines.' },
  { id: 'CLM-0187', claim: 'A new learning app improves exam scores by 25%.', baseScore: 0.86, ecs: 0.74, priority: 'High', status: 'Contradicting', source: 'The available studies report a smaller improvement across different cohorts.' },
  { id: 'CLM-0203', claim: 'Urban air pollution decreased across the district.', baseScore: 0.79, ecs: 0.39, priority: 'Medium', status: 'Supporting', source: 'Several monitoring stations support a downward trend in particulate matter.' },
  { id: 'CLM-0114', claim: 'Daily screen time has no effect on sleep quality.', baseScore: 0.76, ecs: 0.91, priority: 'High', status: 'Conflicting', source: 'Sleep studies point to different effects depending on age and usage patterns.' },
  { id: 'CLM-0221', claim: 'Public transit use increased after fare changes.', baseScore: 0.71, ecs: 0.32, priority: 'Medium', status: 'Supporting', source: 'Transit authority counts show an increase during the first reporting period.' },
  { id: 'CLM-0069', claim: 'The coastal cleanup removed most plastic waste from the shoreline.', baseScore: 0.64, ecs: 0.67, priority: 'Medium', status: 'Mixed evidence', source: 'Cleanup totals are high, but the measured shoreline coverage is incomplete.' },
  { id: 'CLM-0155', claim: 'Remote work leads to higher productivity for every team.', baseScore: 0.58, ecs: 0.88, priority: 'High', status: 'Contradicting', source: 'Productivity results vary substantially by role, team structure, and measurement.' },
  { id: 'CLM-0092', claim: 'The community garden program increased local food access.', baseScore: 0.52, ecs: 0.28, priority: 'Low', status: 'Supporting', source: 'Program participation and distribution records support the claim.' },
  { id: 'CLM-0236', claim: 'Recycling reminders doubled household participation.', baseScore: 0.48, ecs: 0.61, priority: 'Medium', status: 'Conflicting', source: 'Participation rose, although the available sample is smaller than the claim suggests.' },
];

function App() {
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
    setActivePage(label);
    setMobileNavOpen(false);
    if (label !== 'Dashboard' && label !== 'Claims' && label !== 'Model evaluation') {
      setRunMessage(`${label} is not part of this first dashboard prototype yet.`);
    } else {
      setRunMessage('');
    }
  };

  const handleRun = () => {
    if (!claimsFile || !perceptionFile) {
      setRunMessage('Add both valid CSV files before preparing the sample ranking.');
      return;
    }

    setIsPreparingRun(true);
    setRunMessage('');
    window.setTimeout(() => {
      setRunContext({
        objective,
        claimsFilename: claimsFile.name,
        perceptionFilename: perceptionFile.name,
      });
      setSelectedClaim(null);
      setActivePage('Claims');
      setIsPreparingRun(false);
    }, 650);
  };

  return (
    <div className="app-shell">
      <button
        className="mobile-menu-button"
        aria-label="Open navigation"
        onClick={() => setMobileNavOpen((isOpen) => !isOpen)}
      >
        <Menu size={21} />
      </button>
      <aside className={`sidebar ${mobileNavOpen ? 'sidebar--open' : ''}`}>
        <div className="brand">
          <div className="brand-mark">CR</div>
          <div>
            <div className="brand-name">ClaimRank</div>
            <div className="brand-subtitle">AI</div>
          </div>
        </div>
        <div className="sidebar-label">Workspace</div>
        <nav className="sidebar-nav" aria-label="Main navigation">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              className={`nav-item ${(activePage === label || (label === 'Claims' && activePage === 'ClaimDetail')) ? 'nav-item--active' : ''}`}
              key={label}
              onClick={() => handleNavigation(label)}
            >
              <Icon size={18} strokeWidth={(activePage === label || (label === 'Claims' && activePage === 'ClaimDetail')) ? 2.4 : 2} />
              <span>{label}</span>
              {activePage === label && <ChevronRight className="nav-arrow" size={16} />}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="help-icon"><CircleHelp size={17} /></div>
          <div>
            <strong>Need help?</strong>
            <span>Read the documentation</span>
          </div>
        </div>
      </aside>

      <main className="workspace">
        {activePage === 'Model evaluation' ? (
          <ModelEvaluationPage />
        ) : activePage === 'Claims' ? (
          <RankedClaimsPage runContext={runContext} onOpenClaim={(claim) => {
            setSelectedClaim(claim);
            setActivePage('ClaimDetail');
          }} />
        ) : activePage === 'ClaimDetail' && selectedClaim ? (
          <ClaimDetailPage
            claim={selectedClaim}
            runContext={runContext}
            onBack={() => setActivePage('Claims')}
          />
        ) : (
        <>
        <header className="page-header">
          <div>
            <div className="eyebrow">DASHBOARD / NEW RUN</div>
            <h1>New prioritization run</h1>
            <p>Set up a run to rank news claims according to your research objective.</p>
          </div>
          <div className="header-status"><span className="status-dot" /> Prototype mode</div>
        </header>

        <div className="content-grid">
          <section className="main-column">
            <div className="section-heading">
              <div>
                <h2>Upload your datasets</h2>
                <p>Use CSV files with the columns described in the documentation.</p>
              </div>
              <button className="text-button" onClick={() => setRunMessage('CSV templates will be available when the backend is connected.')}>
                View data guide <ChevronRight size={15} />
              </button>
            </div>
            <div className="upload-grid">
              <UploadCard
                title="Claims dataset"
                hint="Your collection of news claims to prioritize."
                file={claimsFile}
                onFileChange={setClaimsFile}
              />
              <UploadCard
                title="User-perception data"
                hint="Ratings that describe how users perceive each claim."
                file={perceptionFile}
                onFileChange={setPerceptionFile}
              />
            </div>

            <section className="panel steps-panel">
              <div className="panel-title-row">
                <div>
                  <h2>What happens next</h2>
                  <p>Your data will move through three transparent stages.</p>
                </div>
                <div className="steps-badge">3 steps</div>
              </div>
              <div className="steps-list">
                <Step number="01" title="Initial ranking" description="Claims are ranked using the selected objective and user-perception signals." />
                <Step number="02" title="Evidence retrieval & ECS" description="Evidence is retrieved and an Evidence Conflict Score is calculated for each claim." />
                <Step number="03" title="Objective-specific ranking" description="The final ranking is adjusted to surface the claims most relevant to your goal." />
              </div>
            </section>

            <section className="objectives-section">
              <div className="section-heading compact">
                <div>
                  <h2>Choose a research objective</h2>
                  <p>This determines how claims are prioritized in your run.</p>
                </div>
              </div>
              <div className="objective-list">
                {objectives.map((item) => (
                  <button
                    className={`objective-card ${objective === item.id ? 'objective-card--selected' : ''}`}
                    key={item.id}
                    onClick={() => setObjective(item.id)}
                  >
                    <span className="radio">{objective === item.id && <span />}</span>
                    <span className="objective-copy">
                      <strong>{item.title}</strong>
                      <span>{item.description}</span>
                    </span>
                    {objective === item.id && <Check className="objective-check" size={18} />}
                  </button>
                ))}
              </div>
            </section>
          </section>

          <aside className="right-column">
            <section className="panel info-panel">
              <div className="info-icon"><Info size={18} /></div>
              <div>
                <h2>About Evidence Conflict Score</h2>
                <p>A high ECS indicates that available evidence conflicts with a claim. It does <strong>not</strong> prove that the claim is false.</p>
                <button className="text-button" onClick={() => setRunMessage('ECS compares retrieved evidence with claim wording; it is not a truth label.')}>Learn more <ChevronRight size={15} /></button>
              </div>
            </section>
            <section className="run-panel">
              <div className="run-panel-heading">
                <div className="run-icon"><BarChart3 size={19} /></div>
                <div>
                  <h2>Ready to prioritize?</h2>
                  <p>Review your selections, then start the prototype run.</p>
                </div>
              </div>
              <div className="run-summary">
                <SummaryRow label="Claims dataset" value={claimsFile ? 'Ready' : 'Not added'} ready={Boolean(claimsFile)} />
                <SummaryRow label="Perception data" value={perceptionFile ? 'Ready' : 'Not added'} ready={Boolean(perceptionFile)} />
                <SummaryRow label="Objective" value={objectives.find((item) => item.id === objective).title} />
              </div>
              <button className="run-button" disabled={!claimsFile || !perceptionFile || isPreparingRun} onClick={handleRun}>
                {isPreparingRun ? 'Preparing sample ranking…' : 'Run prioritization'} {!isPreparingRun && <ChevronRight size={17} />}
              </button>
              {!claimsFile || !perceptionFile ? (
                <div className="run-helper" role="status">Add both valid CSV files to continue.</div>
              ) : null}
              {runMessage && <div className="run-message" role="status">{runMessage}</div>}
            </section>
          </aside>
        </div>
        <footer className="page-footer">
          <span className="footer-dot" /> Illustrative setup — no model inference has run in this prototype.
        </footer>
        </>
        )}
      </main>
    </div>
  );
}

function RankedClaimsPage({ runContext, onOpenClaim }) {
  const [activeTab, setActiveTab] = useState('final');
  const [query, setQuery] = useState('');
  const [evidenceFilter, setEvidenceFilter] = useState('All evidence');
  const [ecsFilter, setEcsFilter] = useState('All ECS levels');

  const filteredClaims = sampleClaims.filter((item) => {
    const normalizedQuery = query.trim().toLowerCase();
    const matchesSearch = !normalizedQuery
      || item.claim.toLowerCase().includes(normalizedQuery)
      || item.id.toLowerCase().includes(normalizedQuery);
    const matchesEvidence = evidenceFilter === 'All evidence' || item.status === evidenceFilter;
    const matchesEcs = ecsFilter === 'All ECS levels'
      || (ecsFilter === 'High' && item.ecs >= 0.7)
      || (ecsFilter === 'Medium' && item.ecs >= 0.4 && item.ecs < 0.7)
      || (ecsFilter === 'Low' && item.ecs < 0.4);
    return matchesSearch && matchesEvidence && matchesEcs;
  });

  const exportCsv = () => {
    const header = ['Rank', 'Claim ID', 'Claim text', 'Base score', 'ECS', 'Priority', 'Evidence status'];
    const rows = filteredClaims.map((item, index) => [
      index + 1, item.id, item.claim, item.baseScore.toFixed(2), item.ecs.toFixed(2), item.priority, item.status,
    ]);
    const csv = [header, ...rows]
      .map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(','))
      .join('\n');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
    link.download = 'claimrank-sample-ranked-claims.csv';
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <>
      <header className="page-header ranked-header">
        <div>
          <div className="eyebrow">02 · RANKED CLAIMS <span className="illustrative-badge">ILLUSTRATIVE</span></div>
          <h1>Ranked claims</h1>
          <p>Run CR-024 · {getObjectiveTitle(runContext?.objective)} · Updated just now (sample state)</p>
        </div>
        <div className="header-status"><span className="status-dot" /> Sample data only</div>
      </header>

      <div className="ranked-content">
        <div className="sample-notice">
          <Info size={15} />
          <span>
            Illustrative sample metrics and scores — no model inference or evidence retrieval has run.
            {runContext && <> Files prepared: <strong>{runContext.claimsFilename}</strong> and <strong>{runContext.perceptionFilename}</strong>.</>}
          </span>
        </div>
        <div className="metrics-grid">
          <MetricCard label="Claims ranked" value="248" detail="illustrative sample claims" />
          <MetricCard label="Mean ECS" value="0.63" detail="illustrative sample metric" />
          <MetricCard label="High conflict" value="42" detail="illustrative sample claims" />
          <MetricCard label="Evidence coverage" value="81%" detail="illustrative sample metric" />
        </div>

        <section className="ranked-card">
          <div className="ranking-toolbar">
            <div className="ranking-tabs" role="tablist" aria-label="Ranking type">
              <button className={activeTab === 'final' ? 'ranking-tab ranking-tab--active' : 'ranking-tab'} onClick={() => setActiveTab('final')} role="tab" aria-selected={activeTab === 'final'}>Final ranking</button>
              <button className={activeTab === 'initial' ? 'ranking-tab ranking-tab--active' : 'ranking-tab'} onClick={() => setActiveTab('initial')} role="tab" aria-selected={activeTab === 'initial'}>Initial ranking</button>
            </div>
            <span className="demo-tab-note">{activeTab === 'initial' ? 'Demo values use the same sample order.' : 'Final ranking · sample state'}</span>
          </div>
          <div className="filter-toolbar">
            <label className="search-field">
              <Search size={16} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search claim text or ID" aria-label="Search claim text or ID" />
              {query && <button aria-label="Clear search" onClick={() => setQuery('')}><X size={14} /></button>}
            </label>
            <select value={evidenceFilter} onChange={(event) => setEvidenceFilter(event.target.value)} aria-label="Filter by evidence status">
              <option>All evidence</option>
              <option>Mixed evidence</option>
              <option>Contradicting</option>
              <option>Conflicting</option>
              <option>Supporting</option>
            </select>
            <select value={ecsFilter} onChange={(event) => setEcsFilter(event.target.value)} aria-label="Filter by ECS level">
              <option>All ECS levels</option>
              <option>High</option>
              <option>Medium</option>
              <option>Low</option>
            </select>
            <button className="export-button" onClick={exportCsv}><Download size={15} /> Export CSV</button>
          </div>
          <div className="table-meta"><span><strong>{filteredClaims.length}</strong> of 248 sample claims</span><span>Page 1 of 25 <ChevronRight size={14} /></span></div>
          <div className="table-scroll">
            <table className="claims-table">
              <thead>
                <tr><th>Rank</th><th>Claim text</th><th>Base score</th><th>ECS</th><th>Priority</th><th>Evidence status</th><th>Action</th></tr>
              </thead>
              <tbody>
                {filteredClaims.map((item, index) => (
                  <tr key={item.id}>
                    <td><span className="rank-number">{String(index + 1).padStart(2, '0')}</span></td>
                    <td><div className="claim-cell"><strong>{item.claim}</strong><span>{item.id} · sample claim</span></div></td>
                    <td><span className="score-value">{item.baseScore.toFixed(2)}</span></td>
                    <td><span className={`ecs-value ecs-${ecsClass(item.ecs)}`}>{item.ecs.toFixed(2)}</span></td>
                    <td><PriorityBadge priority={item.priority} /></td>
                    <td><StatusBadge status={item.status} /></td>
                    <td><button className="open-button" onClick={() => onOpenClaim(item)}>Open <ChevronRight className="open-arrow" size={14} /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            {!filteredClaims.length && <div className="empty-state"><Search size={22} /><strong>No sample claims match</strong><span>Try changing your search or filters.</span></div>}
          </div>
        </section>
        <footer className="page-footer"><span className="footer-dot" /> All claims, scores, and evidence statuses on this page are illustrative sample data.</footer>
      </div>
    </>
  );
}

function MetricCard({ label, value, detail }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function ModelEvaluationPage() {
  const methods = [
    {
      title: 'ECS-only heuristic',
      description: 'Ranks claims using evidence conflict signals alone.',
      score: '0.61',
      bars: [
        { label: 'Ranking alignment', value: 61, color: 'blue' },
        { label: 'Evidence sensitivity', value: 68, color: 'teal' },
        { label: 'Objective fit', value: 53, color: 'purple' },
      ],
    },
    {
      title: 'Combined ranking',
      description: 'Illustrative blend of user perception and evidence signals.',
      score: '0.78',
      bars: [
        { label: 'Ranking alignment', value: 78, color: 'blue' },
        { label: 'Evidence sensitivity', value: 84, color: 'teal' },
        { label: 'Objective fit', value: 76, color: 'purple' },
      ],
    },
  ];

  return (
    <div className="evaluation-page">
      <header className="page-header evaluation-header">
        <div>
          <div className="eyebrow">04 · MODEL EVALUATION <span className="illustrative-badge">ILLUSTRATIVE</span></div>
          <h1>Model evaluation</h1>
          <p>Compare ranking approaches and review responsible quality indicators.</p>
        </div>
        <div className="header-status"><span className="status-dot" /> Prototype data</div>
      </header>

      <div className="evaluation-notice"><Info size={15} /> All scores and metrics on this screen are illustrative prototype values, not measured model results.</div>

      <section className="evaluation-methods">
        <div className="evaluation-section-heading">
          <div><h2>Ranking method comparison</h2><p>Illustrative sample scores across selected quality dimensions.</p></div>
          <span className="sample-pill">Sample comparison</span>
        </div>
        <div className="method-grid">
          {methods.map((method) => (
            <section className="method-card" key={method.title}>
              <div className="method-card-heading">
                <div><h3>{method.title}</h3><p>{method.description}</p></div>
                <strong>{method.score}</strong>
              </div>
              <div className="method-bars">
                {method.bars.map((bar) => (
                  <div className="method-bar-row" key={bar.label}>
                    <div className="method-bar-label"><span>{bar.label}</span><strong>{bar.value}%</strong></div>
                    <div className="method-track"><span className={`method-bar method-bar--${bar.color}`} style={{ width: `${bar.value}%` }} /></div>
                  </div>
                ))}
              </div>
              <span className="method-footnote">Illustrative composite score</span>
            </section>
          ))}
        </div>
      </section>

      <div className="evaluation-lower-grid">
        <section className="evaluation-panel retrieval-panel">
          <div className="evaluation-panel-heading"><div><h2>Evidence retrieval</h2><p>Illustrative retrieval indicators.</p></div><div className="panel-icon"><Search size={17} /></div></div>
          <div className="retrieval-metrics">
            <div><span>Precision@5</span><strong>0.78</strong><small>sample metric</small></div>
            <div><span>Recall@10</span><strong>0.82</strong><small>sample metric</small></div>
            <div><span>Claims with ≥3 sources</span><strong>81%</strong><small>sample coverage</small></div>
          </div>
        </section>

        <section className="evaluation-panel responsible-panel">
          <div className="evaluation-panel-heading"><div><h2>Report model quality responsibly</h2><p>Recommended reporting checklist.</p></div><div className="panel-icon panel-icon--green"><CheckCircle2 size={17} /></div></div>
          <ul className="quality-checklist">
            <li><CheckCircle2 size={15} /> Precision / recall / Macro F1 per objective</li>
            <li><CheckCircle2 size={15} /> Ranking metrics (nDCG@k, Precision@k)</li>
            <li><CheckCircle2 size={15} /> Retrieval coverage and source diversity</li>
            <li><CheckCircle2 size={15} /> Latency, errors, and human-review agreement</li>
          </ul>
        </section>
      </div>
      <footer className="page-footer evaluation-footer"><span className="footer-dot" /> All charts and KPIs on this screen are prototype sample data; replace them with measured evaluation results before reporting.</footer>
    </div>
  );
}

function getObjectiveTitle(objectiveId) {
  return objectives.find((item) => item.id === objectiveId)?.title || 'Remove False Claims';
}

function PriorityBadge({ priority }) {
  return <span className={`priority-badge priority-${priority.toLowerCase()}`}><span />{priority}</span>;
}

function StatusBadge({ status }) {
  return <span className={`status-badge status-${status.toLowerCase().replace(' ', '-')}`}>{status}</span>;
}

function ecsClass(ecs) {
  if (ecs >= 0.7) return 'high';
  if (ecs >= 0.4) return 'medium';
  return 'low';
}

function ClaimDetailPage({ claim, runContext, onBack }) {
  const [actionMessage, setActionMessage] = useState('');
  const [copied, setCopied] = useState(false);
  const [factCheckRequested, setFactCheckRequested] = useState(false);
  const objectiveTitle = getObjectiveTitle(runContext?.objective);

  const copyClaim = async () => {
    try {
      await navigator.clipboard.writeText(claim.claim);
      setCopied(true);
      setActionMessage('Claim text copied to your clipboard.');
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setActionMessage('Copy is unavailable in this browser. Select the claim text manually.');
    }
  };

  return (
    <div className="detail-page">
      <header className="page-header detail-page-header">
        <div>
          <button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Back to Ranked Claims</button>
          <div className="eyebrow">03 · CLAIM INVESTIGATION <span className="illustrative-badge">ILLUSTRATIVE</span></div>
          <h1>Claim detail &amp; evidence investigation</h1>
          <p>Run CR-024 · {claim.id} · {objectiveTitle}</p>
        </div>
        <div className="header-status"><span className="status-dot" /> SAMPLE DATA</div>
      </header>

      <section className="detail-summary-card">
        <div className="detail-card-heading">
          <div><span className="detail-kicker">SELECTED CLAIM · {claim.id}</span><h2>Claim summary</h2></div>
          <StatusBadge status={claim.status} />
        </div>
        <blockquote>“{claim.claim}”</blockquote>
        <div className="claim-meta-grid">
          <div><span>Research objective</span><strong>{objectiveTitle}</strong></div>
          <div><span>Claim ID</span><strong>{claim.id}</strong></div>
          <div><span>Source</span><strong>Sample uploaded claims CSV</strong></div>
        </div>
        <div className="score-summary">
          <div className="score-indicator"><span className="score-ring">{Math.round(claim.baseScore * 100)}%</span><div><strong>Illustrative bias-aware score</strong><span>Percentage-style demo indicator, not a truth probability</span></div></div>
          <div className="score-values">
            <div><span>Bias-aware score</span><strong>{claim.baseScore.toFixed(2)}</strong></div>
            <div><span>Evidence conflict (ECS)</span><strong>{claim.ecs.toFixed(2)}</strong></div>
            <div><span>Final priority (demo index)</span><strong>{(claim.baseScore * (1 + claim.ecs * .15)).toFixed(2)}</strong></div>
          </div>
        </div>
        <p className="illustrative-copy"><Info size={14} /> These values are illustrative demo data for the interface. The final priority is a sample ranking index, not a probability or truth confidence. No claim verification, evidence retrieval, or ML inference has occurred.</p>
      </section>

      <div className="detail-layout">
        <main className="detail-main-column">
          <section className="evidence-section">
            <div className="section-heading">
              <div><h2>Retrieved evidence</h2><p>Mock evidence examples shown for investigation layout only.</p></div>
              <span className="sample-pill">3 sample sources</span>
            </div>
            <div className="evidence-grid">
              <EvidenceInvestigationCard type="Supporting" title="City transport report outlines congestion changes" date="June 12, 2025" summary="A sample municipal report describes a reduction in peak-hour traffic following the policy change." relevance="0.88" />
              <EvidenceInvestigationCard type="Contradicting" title="Independent review questions the comparison baseline" date="July 03, 2025" summary="A mock analysis notes that the before-and-after measurements use different traffic counting periods." relevance="0.79" />
              <EvidenceInvestigationCard type="Neutral" title="Traffic monitoring methodology overview" date="May 28, 2025" summary="This illustrative reference explains how congestion is measured without taking a position on the claim." relevance="0.46" />
            </div>
            <p className="evidence-disclaimer"><Info size={14} /> All source titles, dates, summaries, and relevance scores are mock examples. They have not been retrieved or verified.</p>
          </section>
        </main>

        <aside className="detail-side-column">
          <section className="panel investigation-card verify-card">
            <div className="investigation-card-icon"><ExternalLink size={17} /></div>
            <div><h2>Verify this claim</h2><p>Use FactLLAMA for a future fact-checking workflow. This prototype does not run real verification.</p></div>
            <button className="detail-action-button" onClick={() => { setFactCheckRequested(true); setActionMessage('Fact-checking requires backend integration. No FactLLAMA request was made.'); }}><ExternalLink size={15} /> Fact-check with FactLLAMA</button>
            {factCheckRequested && <p className="action-message" role="status">{actionMessage}</p>}
          </section>
          <section className="panel investigation-card review-card">
            <div className="review-heading"><div className="investigation-card-icon"><CheckCircle2 size={17} /></div><div><h2>Review status</h2><span>Sample investigation state</span></div></div>
            <div className="review-status"><span className="review-dot" /> Awaiting human review</div>
            <p>Evidence has not been retrieved or verified. A reviewer would assess the available sources before making a decision.</p>
          </section>
          <section className="panel investigation-card why-ranked-card">
            <div className="investigation-card-icon"><Info size={17} /></div>
            <h2>Why is this claim highly ranked?</h2>
            <p>This sample claim has a relatively high illustrative bias-aware score and a high ECS, so it is surfaced for investigation under the selected objective.</p>
            <div className="why-ranked-metrics"><span>Sample bias-aware score <strong>{claim.baseScore.toFixed(2)}</strong></span><span>Sample ECS <strong>{claim.ecs.toFixed(2)}</strong></span></div>
          </section>
          <section className="panel detail-actions-card">
            <h2>Claim actions</h2>
            <button className="detail-action-button" onClick={copyClaim}><Copy size={15} /> {copied ? 'Copied' : 'Copy claim text'}</button>
            <button className="detail-action-button detail-action-button--secondary" onClick={() => setActionMessage('Source review is a frontend-only placeholder. Real retrieval requires backend integration.') }><ExternalLink size={15} /> Review sources</button>
            {actionMessage && !factCheckRequested && <p className="action-message" role="status">{actionMessage}</p>}
          </section>
          <div className="detail-side-note"><CheckCircle2 size={16} /><span>All sources, evidence, and scores are illustrative sample data.</span></div>
        </aside>
      </div>
      <footer className="page-footer"><span className="footer-dot" /> Illustrative investigation — no model inference or real evidence retrieval has run.</footer>
    </div>
  );
}

function EvidenceInvestigationCard({ type, title, date, summary, relevance }) {
  return (
    <article className={`investigation-evidence-card evidence-type-${type.toLowerCase()}`}>
      <div className="evidence-card-topline"><span className="evidence-type-badge"><span />{type} evidence</span><span>Sample</span></div>
      <h3>{title}</h3>
      <div className="evidence-date">{date} · Mock publication date</div>
      <p>{summary}</p>
      <div className="relevance-row"><span>Illustrative relevance</span><strong>{relevance}</strong></div>
    </article>
  );
}

function UploadCard({ title, hint, file, onFileChange }) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState('');
  const inputId = title.toLowerCase().replaceAll(' ', '-');

  const acceptFile = (selectedFile) => {
    if (!selectedFile) return;
    if (!selectedFile.name.toLowerCase().endsWith('.csv')) {
      setError('Please choose a CSV file.');
      return;
    }
    if (selectedFile.size > 25 * 1024 * 1024) {
      setError('Please choose a file smaller than 25 MB.');
      return;
    }
    setError('');
    onFileChange(selectedFile);
  };

  return (
    <div
      className={`upload-card ${isDragging ? 'upload-card--dragging' : ''} ${file ? 'upload-card--ready' : ''}`}
      onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        acceptFile(event.dataTransfer.files[0]);
      }}
    >
      <div className="upload-card-header">
        <div className="upload-file-icon"><FileSpreadsheet size={19} /></div>
        <div>
          <h3>{title}</h3>
          <p>{hint}</p>
        </div>
      </div>
      {file ? (
        <div className="selected-file">
          <div className="selected-file-meta">
            <Check size={16} />
            <span>{file.name}</span>
            <small>{formatFileSize(file.size)}</small>
          </div>
          <button aria-label={`Remove ${file.name}`} onClick={() => onFileChange(null)}><X size={16} /></button>
        </div>
      ) : (
        <label className="drop-zone" htmlFor={inputId}>
          <UploadCloud size={21} />
          <span><strong>Choose a CSV</strong> or drag it here</span>
          <small>Maximum file size: 25 MB</small>
          <input id={inputId} type="file" accept=".csv,text/csv" onChange={(event) => acceptFile(event.target.files[0])} />
        </label>
      )}
      {error && <div className="upload-error" role="alert">{error}</div>}
    </div>
  );
}

function Step({ number, title, description }) {
  return (
    <div className="step">
      <div className="step-number">{number}</div>
      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

function SummaryRow({ label, value, ready }) {
  return (
    <div className="summary-row">
      <span>{label}</span>
      <span className={ready ? 'summary-ready' : ''}>{ready && <Check size={13} />}{value}</span>
    </div>
  );
}

function formatFileSize(bytes) {
  if (!bytes) return '0 KB';
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
