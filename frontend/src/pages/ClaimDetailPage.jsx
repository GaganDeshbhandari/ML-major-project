import React, { useEffect, useState } from 'react';
import { ArrowLeft, CheckCircle2, Copy, ExternalLink, Info } from 'lucide-react';
import { getObjectiveTitle } from '../data/sampleData';
import { EvidenceInvestigationCard, StatusBadge, formatMetric, humanize } from '../components/common';
import { ApiError, createFactCheck, getFactCheckJob, getRunClaim } from '../api/client';

function conflictLabel(conflict) {
  if (!conflict) return 'Not scored';
  const labels = {
    computed: 'Computed',
    no_evidence: 'No evidence',
    retrieval_error: 'Retrieval error',
    nli_error: 'NLI error',
    insufficient_stance: 'Insufficient stance',
    disabled: 'Disabled',
  };
  return labels[conflict.status] || humanize(conflict.status);
}

function evidenceCard(record, index, prediction) {
  const item = typeof record === 'string' ? { content: record } : (record || {});
  const stance = prediction?.label;
  const type = stance === 'supports' ? 'Supporting' : stance === 'refutes' ? 'Contradicting' : stance === 'not_enough_info' ? 'Neutral' : 'Evidence';
  const summary = item.content || item.text || item.evidence || 'No summary text was stored for this evidence item.';
  const title = item.title || item.source || `Evidence ${index + 1}`;
  return {
    key: `${index}-${title}`,
    type,
    title,
    date: item.published_at || item.published_date || item.date || null,
    summary: String(summary).length > 420 ? `${String(summary).slice(0, 420)}…` : String(summary),
    relevance: prediction?.confidence ?? item.relevance_score,
    url: item.url,
    source: item.source || (item.url ? 'Linked source' : 'Stored evidence'),
  };
}

export default function ClaimDetailPage({ claim, runContext, onBack, onStartNewRun }) {
  const [detail, setDetail] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [copied, setCopied] = useState(false);
  const [actionMessage, setActionMessage] = useState('');
  const [job, setJob] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobError, setJobError] = useState('');
  const [startingJob, setStartingJob] = useState(false);
  const runId = runContext?.runId || detail?.run_id;
  const entry = detail?.claim || claim;
  const claimId = entry?.claim_id;

  useEffect(() => {
    if (!runId || !claimId) return undefined;
    let cancelled = false;
    setLoadError('');
    getRunClaim(runId, claimId)
      .then((data) => { if (!cancelled) setDetail(data); })
      .catch((failure) => {
        if (cancelled) return;
        if (failure instanceof ApiError && failure.status === 404) {
          setLoadError('This run or claim is no longer available. Run state is held in memory and expires when the API restarts.');
        } else {
          setLoadError(failure.message || 'Unable to load this claim.');
        }
      });
    return () => { cancelled = true; };
  }, [runId, claimId]);

  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const next = await getFactCheckJob(jobId);
        if (cancelled) return;
        setJob(next);
        if (next.status === 'queued' || next.status === 'processing') {
          timer = setTimeout(poll, 1500);
        }
      } catch (failure) {
        if (cancelled) return;
        if (failure instanceof ApiError && failure.status === 404) {
          setJobError('This fact-check job is no longer available. Jobs are held in memory and expire when the API restarts.');
        } else {
          setJobError(failure.message || 'Unable to read the fact-check job.');
        }
        setJob((current) => current ? { ...current, status: 'failed', error: { message: failure.message || 'The job could not be read.' } } : current);
      }
    };
    timer = setTimeout(poll, 1000);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId]);

  if (!entry) {
    return (
      <div className="detail-page">
        <section className="panel state-panel">
          <h2>Claim unavailable</h2>
          <p>No claim was selected. Return to ranked claims and choose a claim to inspect.</p>
          <button className="run-button" onClick={onBack}>Back to Ranked Claims</button>
        </section>
      </div>
    );
  }

  const objectiveTitle = getObjectiveTitle(runContext?.objective || detail?.objective);
  const evidence = Array.isArray(entry.evidence) ? entry.evidence : [];
  const predictions = Array.isArray(entry.conflict?.predictions) ? entry.conflict.predictions : [];
  const predictionByIndex = new Map(predictions.map((prediction) => [prediction.evidence_index, prediction]));
  const evidenceItems = evidence.map((record, index) => evidenceCard(record, index, predictionByIndex.get(index)));
  const factCheckBlocked = entry.retrieval_error
    ? 'Evidence retrieval failed for this claim, so FactLLaMA cannot verify it.'
    : evidence.length === 0
      ? 'No stored evidence is available for this claim, so FactLLaMA cannot verify it.'
      : '';
  const jobRunning = job?.status === 'queued' || job?.status === 'processing';

  const copyClaim = async () => {
    try {
      await navigator.clipboard.writeText(entry.claim);
      setCopied(true);
      setActionMessage('Claim text copied to your clipboard.');
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setActionMessage('Copy is unavailable in this browser. Select the claim text manually.');
    }
  };

  const startFactCheck = async () => {
    if (!runId || !claimId || factCheckBlocked) return;
    setStartingJob(true);
    setJobError('');
    setActionMessage('');
    try {
      const created = await createFactCheck(runId, claimId);
      setJob(created);
      setJobId(created.job_id);
    } catch (failure) {
      setJobError(failure.message || 'Unable to start the FactLLaMA request.');
    } finally {
      setStartingJob(false);
    }
  };

  return (
    <div className="detail-page">
      <header className="page-header detail-page-header">
        <div>
          <button className="back-link" onClick={onBack}><ArrowLeft size={15} /> Back to Ranked Claims</button>
          <div className="eyebrow">03 · CLAIM INVESTIGATION</div>
          <h1>Claim detail &amp; evidence investigation</h1>
          <p>Run {runId || '—'} · {claimId} · {objectiveTitle}</p>
        </div>
        <div className="header-status"><span className="status-dot" /> {conflictLabel(entry.conflict)}</div>
      </header>
      {loadError && <div className="run-message" role="alert">{loadError} <button className="text-button" onClick={onStartNewRun}>Start a new run</button></div>}
      <section className="detail-summary-card">
        <div className="detail-card-heading"><div><span className="detail-kicker">SELECTED CLAIM · {claimId}</span><h2>Claim summary</h2></div><StatusBadge status={conflictLabel(entry.conflict)} /></div>
        <blockquote>“{entry.claim}”</blockquote>
        <div className="claim-meta-grid">
          <div><span>Research objective</span><strong>{objectiveTitle}</strong></div>
          <div><span>Claim ID</span><strong>{claimId}</strong></div>
          <div><span>Predicted truth signal</span><strong>{entry.original?.predicted_truth || '—'}</strong></div>
          <div><span>Predicted TPB signal</span><strong>{entry.original?.predicted_tpb || '—'}</strong></div>
        </div>
        <div className="score-summary">
          <div className="score-indicator"><span className="score-ring">#{entry.final?.rank ?? '—'}</span><div><strong>Final RRF rank</strong><span>Reciprocal-rank fusion of the bias-aware ranking and ECS ranking; not a probability</span></div></div>
          <div className="score-values">
            <div><span>Final RRF score</span><strong>{formatMetric(entry.final?.score, 4)}</strong></div>
            <div><span>Evidence conflict (ECS)</span><strong>{formatMetric(entry.ecs_score, 3)}</strong></div>
            <div><span>Bias-aware rank</span><strong>#{formatMetric(entry.original?.rank)}</strong></div>
            <div><span>ECS rank</span><strong>{entry.ecs_rank === null || entry.ecs_rank === undefined ? '—' : `#${entry.ecs_rank}`}</strong></div>
          </div>
        </div>
        <p className="illustrative-copy"><Info size={14} /> Scores and evidence are read from this completed run. ECS reflects conflict in the retrieved evidence and is not a truth label. No new evidence was retrieved for this page.</p>
      </section>
      <div className="detail-layout">
        <main className="detail-main-column">
          <section className="evidence-section">
            <div className="section-heading">
              <div><h2>Retrieved evidence</h2><p>Evidence stored with the completed prioritization run.</p></div>
              <span className="sample-pill">{evidenceItems.length} source{evidenceItems.length === 1 ? '' : 's'}</span>
            </div>
            {evidenceItems.length > 0 ? (
              <div className="evidence-grid">
                {evidenceItems.map((item) => <EvidenceInvestigationCard key={item.key} {...item} />)}
              </div>
            ) : (
              <div className="empty-state"><Info size={22} /><strong>No stored evidence</strong><span>This claim has no usable evidence for a FactLLaMA request.</span></div>
            )}
            <p className="evidence-disclaimer"><Info size={14} /> Evidence titles, dates, and summaries are shown exactly as stored by the run. Missing fields are marked unavailable rather than filled with sample text.</p>
          </section>
        </main>
        <aside className="detail-side-column">
          <section className="panel investigation-card verify-card">
            <div className="investigation-card-icon"><ExternalLink size={17} /></div>
            <div><h2>Verify this claim</h2><p>FactLLaMA runs only when you explicitly request it. It reuses the stored evidence and does not retrieve new sources.</p></div>
            <button className="detail-action-button" disabled={Boolean(factCheckBlocked) || startingJob || jobRunning} onClick={startFactCheck}>
              <ExternalLink size={15} /> {jobRunning ? 'Fact-check running…' : startingJob ? 'Starting…' : 'Fact-check with FactLLaMA'}
            </button>
            {factCheckBlocked && <p className="action-message" role="status">{factCheckBlocked}</p>}
            {jobError && <p className="action-message action-message--error" role="alert">{jobError}</p>}
            {job && (
              <div className={`factcheck-result factcheck-result--${job.status}`}>
                <div className="factcheck-heading">
                  <strong>FactLLaMA result</strong>
                  <StatusBadge status={humanize(job.status)} />
                </div>
                {(job.status === 'queued' || job.status === 'processing') && <p>Waiting for the model. This can take time on the first run.</p>}
                {job.status === 'completed' && (
                  <>
                    <p className="factcheck-verdict">Verdict: <strong>{job.verdict || 'No verdict returned'}</strong></p>
                    {job.explanation ? <p>{job.explanation}</p> : <p className="state-note">The model returned no structured explanation.</p>}
                    {job.raw_output && <details><summary>Raw model output</summary><pre className="factcheck-output">{job.raw_output}</pre></details>}
                    {job.evidence_references?.length > 0 && <p className="state-note">{job.evidence_references.length} stored evidence reference{job.evidence_references.length === 1 ? '' : 's'} used.</p>}
                    {job.limitations?.length > 0 && <ul className="factcheck-limitations">{job.limitations.map((item) => <li key={item}>{item}</li>)}</ul>}
                  </>
                )}
                {job.status === 'failed' && <p role="alert">{job.error?.message || 'The fact-check failed.'}</p>}
              </div>
            )}
            {actionMessage && <p className="action-message" role="status">{actionMessage}</p>}
          </section>
          <section className="panel investigation-card review-card">
            <div className="review-heading"><div className="investigation-card-icon"><CheckCircle2 size={17} /></div><div><h2>Review status</h2><span>Human review still required</span></div></div>
            <div className="review-status"><span className="review-dot" /> Awaiting human review</div>
            <p>Automated ranking and verdicts are decision support, not verification. A reviewer should assess the stored sources before acting on this claim.</p>
          </section>
          <section className="panel investigation-card why-ranked-card">
            <div className="investigation-card-icon"><Info size={17} /></div>
            <h2>Why is this claim highly ranked?</h2>
            <p>The final rank combines the stored bias-aware rank with the ECS-derived rank. It surfaces claims for investigation under the selected objective; it does not establish truth.</p>
            <div className="why-ranked-metrics">
              <span>Bias-aware rank <strong>#{formatMetric(entry.original?.rank)}</strong></span>
              <span>ECS rank <strong>{entry.ecs_rank === null || entry.ecs_rank === undefined ? '—' : `#${entry.ecs_rank}`}</strong></span>
              <span>Rank change <strong>{formatMetric(entry.rank_change)}</strong></span>
            </div>
          </section>
          <section className="panel detail-actions-card">
            <h2>Claim actions</h2>
            <button className="detail-action-button" onClick={copyClaim}><Copy size={15} /> {copied ? 'Copied' : 'Copy claim text'}</button>
            <button className="detail-action-button detail-action-button--secondary" onClick={onBack}>Back to ranked claims</button>
            <p className="state-note">Run status: {humanize(detail?.status || 'completed')}.</p>
          </section>
        </aside>
      </div>
      <footer className="page-footer"><span className="footer-dot" /> Evidence and scores come from the stored run; FactLLaMA runs only after an explicit request.</footer>
    </div>
  );
}
