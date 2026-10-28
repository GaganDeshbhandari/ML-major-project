import { ChevronRight, Download, Info, Search, X } from 'lucide-react';
import { useState } from 'react';
import { sampleClaims, getObjectiveTitle } from '../data/sampleData';
import { MetricCard, PriorityBadge, StatusBadge, ecsClass } from '../components/common';

export default function RankedClaimsPage({ runContext, onOpenClaim }) {
  const [activeTab, setActiveTab] = useState('final');
  const [query, setQuery] = useState('');
  const [evidenceFilter, setEvidenceFilter] = useState('All evidence');
  const [ecsFilter, setEcsFilter] = useState('All ECS levels');
  const normalizedQuery = query.trim().toLowerCase();
  const filteredClaims = sampleClaims.filter((item) => {
    const matchesSearch = !normalizedQuery || item.claim.toLowerCase().includes(normalizedQuery) || item.id.toLowerCase().includes(normalizedQuery);
    const matchesEvidence = evidenceFilter === 'All evidence' || item.status === evidenceFilter;
    const matchesEcs = ecsFilter === 'All ECS levels' || (ecsFilter === 'High' && item.ecs >= 0.7) || (ecsFilter === 'Medium' && item.ecs >= 0.4 && item.ecs < 0.7) || (ecsFilter === 'Low' && item.ecs < 0.4);
    return matchesSearch && matchesEvidence && matchesEcs;
  });
  const exportCsv = () => {
    const header = ['Rank', 'Claim ID', 'Claim text', 'Base score', 'ECS', 'Priority', 'Evidence status'];
    const rows = filteredClaims.map((item, index) => [index + 1, item.id, item.claim, item.baseScore.toFixed(2), item.ecs.toFixed(2), item.priority, item.status]);
    const csv = [header, ...rows].map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\n');
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' }));
    link.download = 'claimrank-sample-ranked-claims.csv'; link.click(); URL.revokeObjectURL(link.href);
  };
  return (
    <>
      <header className="page-header ranked-header"><div><div className="eyebrow">02 · RANKED CLAIMS <span className="illustrative-badge">ILLUSTRATIVE</span></div><h1>Ranked claims</h1><p>Run CR-024 · {getObjectiveTitle(runContext?.objective)} · Updated just now (sample state)</p></div><div className="header-status"><span className="status-dot" /> Sample data only</div></header>
      <div className="ranked-content">
        <div className="sample-notice"><Info size={15} /><span>Illustrative sample metrics and scores — no model inference or evidence retrieval has run.{runContext && <> Files prepared: <strong>{runContext.claimsFilename}</strong> and <strong>{runContext.perceptionFilename}</strong>.</>}</span></div>
        <div className="metrics-grid"><MetricCard label="Claims ranked" value="248" detail="illustrative sample claims" /><MetricCard label="Mean ECS" value="0.63" detail="illustrative sample metric" /><MetricCard label="High conflict" value="42" detail="illustrative sample claims" /><MetricCard label="Evidence coverage" value="81%" detail="illustrative sample metric" /></div>
        <section className="ranked-card">
          <div className="ranking-toolbar"><div className="ranking-tabs" role="tablist" aria-label="Ranking type"><button className={activeTab === 'final' ? 'ranking-tab ranking-tab--active' : 'ranking-tab'} onClick={() => setActiveTab('final')} role="tab" aria-selected={activeTab === 'final'}>Final ranking</button><button className={activeTab === 'initial' ? 'ranking-tab ranking-tab--active' : 'ranking-tab'} onClick={() => setActiveTab('initial')} role="tab" aria-selected={activeTab === 'initial'}>Initial ranking</button></div><span className="demo-tab-note">{activeTab === 'initial' ? 'Demo values use the same sample order.' : 'Final ranking · sample state'}</span></div>
          <div className="filter-toolbar"><label className="search-field"><Search size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search claim text or ID" aria-label="Search claim text or ID" />{query && <button aria-label="Clear search" onClick={() => setQuery('')}><X size={14} /></button>}</label><select value={evidenceFilter} onChange={(event) => setEvidenceFilter(event.target.value)} aria-label="Filter by evidence status"><option>All evidence</option><option>Mixed evidence</option><option>Contradicting</option><option>Conflicting</option><option>Supporting</option></select><select value={ecsFilter} onChange={(event) => setEcsFilter(event.target.value)} aria-label="Filter by ECS level"><option>All ECS levels</option><option>High</option><option>Medium</option><option>Low</option></select><button className="export-button" onClick={exportCsv}><Download size={15} /> Export CSV</button></div>
          <div className="table-meta"><span><strong>{filteredClaims.length}</strong> of 248 sample claims</span><span>Page 1 of 25 <ChevronRight size={14} /></span></div>
          <div className="table-scroll"><table className="claims-table"><thead><tr><th>Rank</th><th>Claim text</th><th>Base score</th><th>ECS</th><th>Priority</th><th>Evidence status</th><th>Action</th></tr></thead><tbody>{filteredClaims.map((item, index) => <tr key={item.id}><td><span className="rank-number">{String(index + 1).padStart(2, '0')}</span></td><td><div className="claim-cell"><strong>{item.claim}</strong><span>{item.id} · sample claim</span></div></td><td><span className="score-value">{item.baseScore.toFixed(2)}</span></td><td><span className={`ecs-value ecs-${ecsClass(item.ecs)}`}>{item.ecs.toFixed(2)}</span></td><td><PriorityBadge priority={item.priority} /></td><td><StatusBadge status={item.status} /></td><td><button className="open-button" onClick={() => onOpenClaim(item)}>Open <ChevronRight className="open-arrow" size={14} /></button></td></tr>)}</tbody></table>{!filteredClaims.length && <div className="empty-state"><Search size={22} /><strong>No sample claims match</strong><span>Try changing your search or filters.</span></div>}</div>
        </section>
        <footer className="page-footer"><span className="footer-dot" /> All claims, scores, and evidence statuses on this page are illustrative sample data.</footer>
      </div>
    </>
  );
}
