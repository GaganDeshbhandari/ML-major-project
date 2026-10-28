import { CheckCircle2, Info, Search } from 'lucide-react';

const methods = [
  { title: 'ECS-only heuristic', description: 'Ranks claims using evidence conflict signals alone.', score: '0.61', bars: [['Ranking alignment', 61, 'blue'], ['Evidence sensitivity', 68, 'teal'], ['Objective fit', 53, 'purple']] },
  { title: 'Combined ranking', description: 'Illustrative blend of user perception and evidence signals.', score: '0.78', bars: [['Ranking alignment', 78, 'blue'], ['Evidence sensitivity', 84, 'teal'], ['Objective fit', 76, 'purple']] },
];

export default function ModelEvaluationPage() {
  return (
    <div className="evaluation-page">
      <header className="page-header evaluation-header"><div><div className="eyebrow">04 · MODEL EVALUATION <span className="illustrative-badge">ILLUSTRATIVE</span></div><h1>Model evaluation</h1><p>Compare ranking approaches and review responsible quality indicators.</p></div><div className="header-status"><span className="status-dot" /> Prototype data</div></header>
      <div className="evaluation-notice"><Info size={15} /> All scores and metrics on this screen are illustrative prototype values, not measured model results.</div>
      <section className="evaluation-methods"><div className="evaluation-section-heading"><div><h2>Ranking method comparison</h2><p>Illustrative sample scores across selected quality dimensions.</p></div><span className="sample-pill">Sample comparison</span></div><div className="method-grid">{methods.map((method) => <section className="method-card" key={method.title}><div className="method-card-heading"><div><h3>{method.title}</h3><p>{method.description}</p></div><strong>{method.score}</strong></div><div className="method-bars">{method.bars.map(([label, value, color]) => <div className="method-bar-row" key={label}><div className="method-bar-label"><span>{label}</span><strong>{value}%</strong></div><div className="method-track"><span className={`method-bar method-bar--${color}`} style={{ width: `${value}%` }} /></div></div>)}</div><span className="method-footnote">Illustrative composite score</span></section>)}</div></section>
      <div className="evaluation-lower-grid">
        <section className="evaluation-panel retrieval-panel"><div className="evaluation-panel-heading"><div><h2>Evidence retrieval</h2><p>Illustrative retrieval indicators.</p></div><div className="panel-icon"><Search size={17} /></div></div><div className="retrieval-metrics"><div><span>Precision@5</span><strong>0.78</strong><small>sample metric</small></div><div><span>Recall@10</span><strong>0.82</strong><small>sample metric</small></div><div><span>Claims with ≥3 sources</span><strong>81%</strong><small>sample coverage</small></div></div></section>
        <section className="evaluation-panel responsible-panel"><div className="evaluation-panel-heading"><div><h2>Report model quality responsibly</h2><p>Recommended reporting checklist.</p></div><div className="panel-icon panel-icon--green"><CheckCircle2 size={17} /></div></div><ul className="quality-checklist"><li><CheckCircle2 size={15} /> Precision / recall / Macro F1 per objective</li><li><CheckCircle2 size={15} /> Ranking metrics (nDCG@k, Precision@k)</li><li><CheckCircle2 size={15} /> Retrieval coverage and source diversity</li><li><CheckCircle2 size={15} /> Latency, errors, and human-review agreement</li></ul></section>
      </div>
      <footer className="page-footer evaluation-footer"><span className="footer-dot" /> All charts and KPIs on this screen are prototype sample data; replace them with measured evaluation results before reporting.</footer>
    </div>
  );
}
