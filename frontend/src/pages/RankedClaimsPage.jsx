import React, { useEffect, useState } from 'react';
import { ChevronRight, Download, Info, RefreshCcw, Search, X } from 'lucide-react';
import { getObjectiveTitle } from '../data/sampleData';
import {
  MetricCard,
  StatusBadge,
  ecsClass,
  formatDateTime,
  formatMetric,
  formatPercent,
  humanize,
} from '../components/common';
import {
  ApiError,
  downloadRunExport,
  getRunStatus,
  getRunSummary,
  listRunClaims,
} from '../api/client';

const PAGE_SIZE = 10;

const EVIDENCE_OPTIONS = [
  ['all', 'All evidence'],
  ['with', 'Has evidence'],
  ['without', 'No evidence'],
  ['computed', 'Computed conflict'],
  ['insufficient_stance', 'Insufficient stance'],
  ['retrieval_error', 'Retrieval error'],
  ['nli_error', 'NLI error'],
];

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

function prioritySignal(item, objective) {
  if (objective === 'reduce_user_disagreement') {
    return <span className="score-value" title="Disputability from the bias-aware ranking">{formatMetric(item.original?.disputability, 2)}</span>;
  }
  const value = objective === 'correct_user_misconceptions'
    ? item.original?.predicted_tpb
    : item.original?.predicted_truth;
  if (!value) return <span className="muted-value">—</span>;
  return <span title="Stored bias-aware model signal"><StatusBadge status={value} /></span>;
}

function resolvedFilters(evidenceFilter, ecsFilter) {
  const filters = {};
  if (evidenceFilter === 'with') filters.has_evidence = true;
  else if (evidenceFilter === 'without') filters.has_evidence = false;
  else if (evidenceFilter !== 'all') filters.conflict_status = evidenceFilter;
  if (ecsFilter !== 'all') filters.conflict = ecsFilter;
  return filters;
}

function phaseStatus(phase, runStatus) {
  if (phase === 'processing') return humanize(runStatus?.stage) || 'Processing';
  return humanize(phase);
}

export default function RankedClaimsPage({ runContext, onOpenClaim, onStartNewRun }) {
  const runId = runContext?.runId;
  const [phase, setPhase] = useState(runId ? 'loading' : 'empty');
  const [runStatus, setRunStatus] = useState(null);
  const [summary, setSummary] = useState(null);
  const [summaryError, setSummaryError] = useState('');
  const [claims, setClaims] = useState(null);
  const [claimsLoading, setClaimsLoading] = useState(false);
  const [claimsError, setClaimsError] = useState('');
  const [error, setError] = useState('');
  const [exportError, setExportError] = useState('');
  const [activeTab, setActiveTab] = useState('final');
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [evidenceFilter, setEvidenceFilter] = useState('all');
  const [ecsFilter, setEcsFilter] = useState('all');
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(query.trim()), 300);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    if (!runId) {
      setPhase('empty');
      return undefined;
    }
    let cancelled = false;
    let timer;
    setPhase('loading');
    setError('');
    setSummary(null);
    setClaims(null);

    const poll = async () => {
      try {
        const status = await getRunStatus(runId);
        if (cancelled) return;
        setRunStatus(status);
        if (status.status === 'completed') {
          setPhase('completed');
        } else if (status.status === 'failed') {
          setError(status.error?.message || 'The prioritization run failed.');
          setPhase('failed');
        } else {
          setPhase('processing');
          timer = setTimeout(poll, 1500);
        }
      } catch (failure) {
        if (cancelled) return;
        if (failure instanceof ApiError && failure.status === 404) {
          setError('This run is no longer available. Run state is held in memory and is lost when the API restarts.');
          setPhase('expired');
        } else {
          setError(failure.message || 'Unable to read the run status.');
          setPhase('error');
        }
      }
    };

    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runId]);

  useEffect(() => {
    if (phase !== 'completed' || !runId) return undefined;
    let cancelled = false;
    setSummaryError('');
    getRunSummary(runId)
      .then((data) => { if (!cancelled) setSummary(data); })
      .catch((failure) => {
        if (cancelled) return;
        if (failure instanceof ApiError && failure.status === 404) {
          setError('This run is no longer available. Start a new run to continue.');
          setPhase('expired');
        } else {
          setSummaryError(failure.message || 'Unable to load the run summary.');
        }
      });
    return () => { cancelled = true; };
  }, [phase, runId]);

  useEffect(() => {
    if (phase !== 'completed' || !runId) return undefined;
    let cancelled = false;
    setClaimsLoading(true);
    setClaimsError('');
    listRunClaims(runId, {
      q: debouncedQuery || undefined,
      ranking: activeTab,
      offset,
      limit: PAGE_SIZE,
      ...resolvedFilters(evidenceFilter, ecsFilter),
    })
      .then((data) => { if (!cancelled) setClaims(data); })
      .catch((failure) => {
        if (cancelled) return;
        if (failure instanceof ApiError && failure.status === 404) {
          setError('This run is no longer available. Start a new run to continue.');
          setPhase('expired');
        } else {
          setClaimsError(failure.message || 'Unable to load ranked claims.');
        }
      })
      .finally(() => { if (!cancelled) setClaimsLoading(false); });
    return () => { cancelled = true; };
  }, [phase, runId, debouncedQuery, activeTab, evidenceFilter, ecsFilter, offset]);

  const objectiveTitle = getObjectiveTitle(runContext?.objective);
  const objectiveKey = runContext?.objective;
  const summaryClaims = summary?.total_claims ?? claims?.total_claims;
  const matchedClaims = claims?.matched_claims ?? 0;
  const pageNumber = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(matchedClaims / PAGE_SIZE));
  const retrievalErrors = summary?.component_error_counts?.retrieval || 0;
  const ecsErrors = summary?.component_error_counts?.ecs || 0;

  const changeFilter = (setter) => (event) => {
    setter(event.target.value);
    setOffset(0);
  };

  const exportCsv = async () => {
    setExportError('');
    try {
      await downloadRunExport(runId);
    } catch (failure) {
      setExportError(failure.message || 'Unable to export the run.');
    }
  };

  const header = (
    <header className="page-header ranked-header">
      <div>
        <div className="eyebrow">02 · RANKED CLAIMS</div>
        <h1>Ranked claims</h1>
        <p>
          {runContext
            ? `Run ${runId} · ${objectiveTitle} · ${runContext.claimCount} input claims`
            : 'Start a prioritization run from the dashboard.'}
        </p>
      </div>
      <div className="header-status"><span className="status-dot" /> {phaseStatus(phase, runStatus)}</div>
    </header>
  );

  if (phase === 'empty') {
    return (
      <>
        {header}
        <div className="ranked-content">
          <section className="panel state-panel">
            <div className="state-icon"><Info size={20} /></div>
            <h2>No active run</h2>
            <p>Upload a claims CSV and a user-perception CSV, then start a prioritization run. Only then will ranked results appear here.</p>
            <button className="run-button" onClick={onStartNewRun}>Go to the dashboard <ChevronRight size={17} /></button>
          </section>
        </div>
      </>
    );
  }

  if (phase === 'loading' || phase === 'processing') {
    const progress = Math.max(0, Math.min(1, Number(runStatus?.progress) || 0));
    return (
      <>
        {header}
        <div className="ranked-content">
          <section className="panel state-panel">
            <div className="state-icon"><RefreshCcw size={20} /></div>
            <h2>{phase === 'loading' ? 'Checking run status…' : 'Preparing your ranking'}</h2>
            <p>
              {phase === 'loading'
                ? 'Contacting the API for this run.'
                : `${humanize(runStatus?.status)} · stage: ${humanize(runStatus?.stage)}`}
            </p>
            <div className="method-track"><span className="method-bar method-bar--blue" style={{ width: `${Math.round(progress * 100)}%` }} /></div>
            <span className="state-progress">{Math.round(progress * 100)}%</span>
            <p className="state-note">This page polls the API. Run state is held in memory and may expire if the API restarts.</p>
          </section>
        </div>
      </>
    );
  }

  if (phase === 'failed' || phase === 'expired' || phase === 'error') {
    return (
      <>
        {header}
        <div className="ranked-content">
          <section className="panel state-panel">
            <div className="state-icon"><Info size={20} /></div>
            <h2>{phase === 'expired' ? 'Run unavailable' : phase === 'failed' ? 'Run failed' : 'Unable to load this run'}</h2>
            <p>{error || 'The API could not return this run.'}</p>
            {runStatus?.error?.stage && <p className="state-note">Failed stage: {humanize(runStatus.error.stage)}</p>}
            <button className="run-button" onClick={onStartNewRun}>Start a new run <ChevronRight size={17} /></button>
          </section>
        </div>
      </>
    );
  }

  return (
    <>
      {header}
      <div className="ranked-content">
        <div className="sample-notice">
          <Info size={15} />
          <span>
            Loaded from run <strong>{runId}</strong> completed {formatDateTime(runStatus?.completed_at)}. Scores are model outputs, not verified facts.
            {' '}Export downloads the API's stored results.
          </span>
        </div>
        {retrievalErrors + ecsErrors > 0 && (
          <div className="sample-notice sample-notice--warning">
            <Info size={15} />
            <span>
              {retrievalErrors} claim{retrievalErrors === 1 ? '' : 's'} had retrieval errors and {ecsErrors} had ECS errors.
              {' '}Affected values are shown as “—”, not zero.
            </span>
          </div>
        )}
        {summaryError && <div className="run-message" role="alert">{summaryError}</div>}
        {exportError && <div className="run-message" role="alert">{exportError}</div>}
        <div className="metrics-grid">
          <MetricCard label="Claims ranked" value={formatMetric(summaryClaims)} detail="from this completed run" />
          <MetricCard label="Mean ECS" value={formatMetric(summary?.score_summaries?.ecs_score?.mean, 3)} detail="available ECS values only" />
          <MetricCard label="High conflict" value={formatMetric(summary?.high_conflict_claim_count)} detail={`ECS ≥ ${summary?.high_conflict_threshold ?? 0.7}`} />
          <MetricCard label="Evidence coverage" value={formatPercent(summary?.evidence_coverage)} detail="claims with stored evidence" />
        </div>
        <section className="ranked-card">
          <div className="ranking-toolbar">
            <div className="ranking-tabs" role="tablist" aria-label="Ranking type">
              <button className={activeTab === 'final' ? 'ranking-tab ranking-tab--active' : 'ranking-tab'} onClick={() => { setActiveTab('final'); setOffset(0); }} role="tab" aria-selected={activeTab === 'final'}>Final ranking</button>
              <button className={activeTab === 'initial' ? 'ranking-tab ranking-tab--active' : 'ranking-tab'} onClick={() => { setActiveTab('initial'); setOffset(0); }} role="tab" aria-selected={activeTab === 'initial'}>Initial ranking</button>
            </div>
            <span className="demo-tab-note">{activeTab === 'initial' ? 'Sorted by bias-aware original rank' : `Final ranking · RRF k=${summary?.rrf_k ?? '—'}`}</span>
          </div>
          <div className="filter-toolbar">
            <label className="search-field">
              <Search size={16} />
              <input value={query} onChange={(event) => { setQuery(event.target.value); setOffset(0); }} placeholder="Search claim text or ID" aria-label="Search claim text or ID" />
              {query && <button aria-label="Clear search" onClick={() => { setQuery(''); setOffset(0); }}><X size={14} /></button>}
            </label>
            <select value={evidenceFilter} onChange={changeFilter(setEvidenceFilter)} aria-label="Filter by evidence status">
              {EVIDENCE_OPTIONS.map(([value, label]) => <option value={value} key={value}>{label}</option>)}
            </select>
            <select value={ecsFilter} onChange={changeFilter(setEcsFilter)} aria-label="Filter by ECS level">
              <option value="all">All ECS levels</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <button className="export-button" onClick={exportCsv}><Download size={15} /> Export CSV</button>
          </div>
          <div className="table-meta">
            <span><strong>{matchedClaims}</strong> of {summaryClaims ?? '—'} claims</span>
            <span className="pagination">
              <button className="text-button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous</button>
              <span>Page {pageNumber} of {totalPages}</span>
              <button className="text-button" disabled={!claims?.has_more} onClick={() => setOffset(offset + PAGE_SIZE)}>Next <ChevronRight size={14} /></button>
            </span>
          </div>
          <div className="table-scroll">
            {claimsLoading ? (
              <div className="empty-state"><RefreshCcw size={22} /><strong>Loading ranked claims…</strong><span>Reading stored results from the API.</span></div>
            ) : claimsError ? (
              <div className="empty-state"><Info size={22} /><strong>Could not load claims</strong><span>{claimsError}</span></div>
            ) : (
              <>
                <table className="claims-table">
                  <thead><tr><th>Rank</th><th>Claim text</th><th>Final score</th><th>ECS</th><th>Priority signal</th><th>Evidence status</th><th>Action</th></tr></thead>
                  <tbody>
                    {(claims?.claims || []).map((item) => {
                      const rank = activeTab === 'initial' ? item.original?.rank : item.final?.rank;
                      return (
                        <tr key={item.claim_id}>
                          <td><span className="rank-number">{rank === null || rank === undefined ? '—' : String(rank).padStart(2, '0')}</span></td>
                          <td><div className="claim-cell"><strong>{item.claim}</strong><span>{item.claim_id} · rank change {formatMetric(item.rank_change)}</span></div></td>
                          <td><span className="score-value">{formatMetric(item.final?.score, 4)}</span></td>
                          <td>{item.ecs_score === null || item.ecs_score === undefined ? <span className="muted-value">—</span> : <span className={`ecs-value ecs-${ecsClass(item.ecs_score)}`}>{formatMetric(item.ecs_score, 2)}</span>}</td>
                          <td>{prioritySignal(item, objectiveKey)}</td>
                          <td><StatusBadge status={conflictLabel(item.conflict)} /></td>
                          <td><button className="open-button" onClick={() => onOpenClaim(item)}>Open <ChevronRight className="open-arrow" size={14} /></button></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                {!claims?.claims?.length && <div className="empty-state"><Search size={22} /><strong>No claims match</strong><span>Try changing your search or filters.</span></div>}
              </>
            )}
          </div>
        </section>
        <footer className="page-footer"><span className="footer-dot" /> Ranking values and evidence are loaded from the completed run; ECS is not a truth label.</footer>
      </div>
    </>
  );
}
