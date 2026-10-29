import React from 'react';
import { BarChart3, Check, ChevronRight, Info } from 'lucide-react';
import { objectives } from '../data/sampleData';
import { Step, SummaryRow, UploadCard } from '../components/common';

const DATA_GUIDE = 'Claims CSV columns: claim_id, claim. Perception CSV columns: claim_id, rating. Ratings: True, Mostly True, Mixture, Mostly False, False.';

function validationLabel(file, validation) {
  if (!file) return 'Not added';
  if (!validation) return 'Not checked';
  if (validation.status === 'checking') return 'Checking…';
  if (validation.status === 'valid') return `Valid · ${validation.validation.row_count} rows`;
  if (validation.status === 'invalid') return 'Needs fixes';
  return 'Validation error';
}

function PreviewTable({ title, preview }) {
  if (!preview) return null;
  return (
    <div className="preview-block">
      <div className="preview-heading"><strong>{title}</strong><span>{preview.row_count} rows · {preview.columns.length} columns</span></div>
      <div className="preview-scroll">
        <table className="preview-table">
          <thead><tr>{preview.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>
            {(preview.preview || []).map((row, index) => (
              <tr key={`${title}-${index}`}>
                {preview.columns.map((column) => <td key={column}>{row[column] || '—'}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function DashboardPage({
  claimsFile, perceptionFile, objective, setClaimsFile, setPerceptionFile,
  setObjective, runMessage, setRunMessage, isPreparingRun, handleRun,
  claimsValidation, perceptionValidation, claimsPreview, perceptionPreview,
}) {
  const claimsReady = claimsValidation?.status === 'valid';
  const perceptionReady = perceptionValidation?.status === 'valid';
  const hasPreview = Boolean(claimsPreview || perceptionPreview);

  return (
    <>
      <header className="page-header">
        <div><div className="eyebrow">DASHBOARD / NEW RUN</div><h1>New prioritization run</h1><p>Set up a run to rank news claims according to your research objective.</p></div>
        <div className="header-status"><span className="status-dot" /> API-backed workflow</div>
      </header>
      <div className="content-grid">
        <section className="main-column">
          <div className="section-heading">
            <div><h2>Upload your datasets</h2><p>CSV files are validated by the API before a run can start.</p></div>
            <button className="text-button" onClick={() => setRunMessage(DATA_GUIDE)}>View data guide <ChevronRight size={15} /></button>
          </div>
          <div className="upload-grid">
            <UploadCard title="Claims dataset" hint="Your collection of news claims to prioritize." file={claimsFile} onFileChange={setClaimsFile} validation={claimsValidation} />
            <UploadCard title="User-perception data" hint="Ratings that describe how users perceive each claim." file={perceptionFile} onFileChange={setPerceptionFile} validation={perceptionValidation} />
          </div>
          {hasPreview && (
            <section className="panel preview-panel">
              <div className="panel-title-row"><div><h2>CSV preview</h2><p>The first rows returned by the validation API.</p></div><span className="steps-badge">Preview</span></div>
              <div className="preview-grid">
                <PreviewTable title="Claims" preview={claimsPreview} />
                <PreviewTable title="User perception" preview={perceptionPreview} />
              </div>
            </section>
          )}
          <section className="panel steps-panel">
            <div className="panel-title-row"><div><h2>What happens next</h2><p>Your data will move through three transparent stages.</p></div><div className="steps-badge">3 steps</div></div>
            <div className="steps-list">
              <Step number="01" title="Initial ranking" description="Claims are ranked using the selected objective and user-perception signals." />
              <Step number="02" title="Evidence retrieval & ECS" description="Evidence is retrieved and an Evidence Conflict Score is calculated for each claim." />
              <Step number="03" title="Objective-specific ranking" description="The final ranking is adjusted to surface the claims most relevant to your goal." />
            </div>
          </section>
          <section className="objectives-section">
            <div className="section-heading compact"><div><h2>Choose a research objective</h2><p>This determines how claims are prioritized in your run.</p></div></div>
            <div className="objective-list">
              {objectives.map((item) => (
                <button className={`objective-card ${objective === item.id ? 'objective-card--selected' : ''}`} key={item.id} onClick={() => setObjective(item.id)}>
                  <span className="radio">{objective === item.id && <span />}</span>
                  <span className="objective-copy"><strong>{item.title}</strong><span>{item.description}</span></span>
                  {objective === item.id && <Check className="objective-check" size={18} />}
                </button>
              ))}
            </div>
          </section>
        </section>
        <aside className="right-column">
          <section className="panel info-panel"><div className="info-icon"><Info size={18} /></div><div><h2>About Evidence Conflict Score</h2><p>A high ECS indicates that available evidence conflicts with a claim. It does <strong>not</strong> prove that the claim is false.</p><button className="text-button" onClick={() => setRunMessage('ECS compares retrieved evidence with claim wording; it is not a truth label.')}>Learn more <ChevronRight size={15} /></button></div></section>
          <section className="run-panel">
            <div className="run-panel-heading"><div className="run-icon"><BarChart3 size={19} /></div><div><h2>Ready to prioritize?</h2><p>Review your selections, then start the run.</p></div></div>
            <div className="run-summary">
              <SummaryRow label="Claims dataset" value={validationLabel(claimsFile, claimsValidation)} ready={claimsReady} />
              <SummaryRow label="Perception data" value={validationLabel(perceptionFile, perceptionValidation)} ready={perceptionReady} />
              <SummaryRow label="Objective" value={objectives.find((item) => item.id === objective).title} />
            </div>
            <button className="run-button" disabled={!claimsReady || !perceptionReady || isPreparingRun} onClick={handleRun}>{isPreparingRun ? 'Starting run…' : 'Run prioritization'} {!isPreparingRun && <ChevronRight size={17} />}</button>
            {!claimsReady || !perceptionReady ? <div className="run-helper" role="status">Add both valid CSV files to continue.</div> : null}
            {runMessage && <div className="run-message" role="status">{runMessage}</div>}
          </section>
        </aside>
      </div>
      <footer className="page-footer"><span className="footer-dot" /> Runs execute on the local API. Evidence retrieval and model inference start only after you submit a run.</footer>
    </>
  );
}
