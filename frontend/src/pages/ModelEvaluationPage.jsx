import React, { useEffect, useState } from 'react';
import { CheckCircle2, Info, RefreshCcw, Search } from 'lucide-react';
import { StatusBadge, formatDateTime, formatMetric, humanize } from '../components/common';
import {
  API_BASE_URL,
  getEvaluationSummary,
  getPrioritizationComparison,
  getRetrievalMetrics,
} from '../api/client';

const EXPERIMENT_LABELS = {
  factllama_fact_verification: 'FactLLaMA fact verification',
  nli_stance_classification: 'NLI stance classification',
  claim_ranking_prioritization: 'Claim-ranking prioritization',
  evidence_retrieval: 'Evidence retrieval',
};

const METHOD_ORDER = ['baseline', 'ecs_only', 'fused'];

function metricRows(experiment) {
  const metrics = experiment?.metrics || {};
  if (experiment?.name === 'factllama_fact_verification') {
    return [
      ['Accuracy', metrics.accuracy],
      ['Macro F1', metrics.macro_f1],
      ['Scored examples', experiment.counts?.scored],
    ];
  }
  if (experiment?.name === 'nli_stance_classification') {
    return [
      ['Accuracy', metrics.accuracy],
      ['Macro F1', metrics.macro_f1],
      ['Coverage', metrics.coverage],
      ['Evaluated examples', metrics.evaluated_examples],
    ];
  }
  if (experiment?.name === 'claim_ranking_prioritization') {
    return [
      ['Objectives evaluated', metrics && typeof metrics === 'object' ? Object.keys(metrics).length : null],
      ['Claims evaluated', experiment.dataset?.num_claims_evaluated],
    ];
  }
  return Object.entries(metrics).filter(([, value]) => typeof value !== 'object');
}

function MetricRows({ metrics }) {
  const rows = Object.entries(metrics || {}).filter(([, value]) => value === null || typeof value !== 'object');
  if (!rows.length) return <p className="method-unavailable">No metric values reported.</p>;
  return (
    <div className="method-bars">
      {rows.map(([label, value]) => (
        <div className="method-bar-row" key={label}>
          <div className="method-bar-label"><span>{label}</span><strong>{formatMetric(value, 4)}</strong></div>
          {typeof value === 'number' && value >= 0 && value <= 1 && (
            <div className="method-track"><span className="method-bar method-bar--blue" style={{ width: `${value * 100}%` }} /></div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function ModelEvaluationPage() {
  const [state, setState] = useState({
    loading: true,
    summary: null,
    comparison: null,
    retrieval: null,
    summaryError: '',
    comparisonError: '',
    retrievalError: '',
  });
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState((previous) => ({ ...previous, loading: true }));
    Promise.allSettled([
      getEvaluationSummary(),
      getPrioritizationComparison(),
      getRetrievalMetrics(),
    ]).then(([summary, comparison, retrieval]) => {
      if (cancelled) return;
      const failure = (result) => result.status === 'rejected' ? (result.reason?.message || 'Request failed.') : '';
      setState({
        loading: false,
        summary: summary.status === 'fulfilled' ? summary.value : null,
        comparison: comparison.status === 'fulfilled' ? comparison.value : null,
        retrieval: retrieval.status === 'fulfilled' ? retrieval.value : null,
        summaryError: failure(summary),
        comparisonError: failure(comparison),
        retrievalError: failure(retrieval),
      });
    });
    return () => { cancelled = true; };
  }, [reloadKey]);

  const { loading, summary, comparison, retrieval, summaryError, comparisonError, retrievalError } = state;
  const allFailed = !loading && !summary && !comparison && !retrieval;
  const experiments = summary?.experiments || [];
  const comparisonMethods = comparison?.methods || {};
  const orderedMethods = [
    ...METHOD_ORDER.filter((key) => comparisonMethods[key]).map((key) => [key, comparisonMethods[key]]),
    ...Object.entries(comparisonMethods).filter(([key]) => !METHOD_ORDER.includes(key)),
  ];
  const retrievalMetrics = retrieval?.metrics ? Object.entries(retrieval.metrics) : [];
  const expectedRetrievalMetrics = retrieval?.expected_metrics || ['Recall@K', 'Precision@K', 'MRR'];

  return (
    <div className="evaluation-page">
      <header className="page-header evaluation-header">
        <div>
          <div className="eyebrow">04 · MODEL EVALUATION</div>
          <h1>Model evaluation</h1>
          <p>Read saved evaluation artifacts through the API. No training or inference runs on this page.</p>
        </div>
        <div className="header-status"><span className="status-dot" /> {loading ? 'Loading' : summary ? humanize(summary.status) : 'Unavailable'}</div>
      </header>
      <div className={`evaluation-notice ${allFailed ? 'evaluation-notice--error' : ''}`}>
        <Info size={15} />
        <span>
          {loading
            ? `Loading saved metrics from ${API_BASE_URL}…`
            : allFailed
              ? `Could not reach the evaluation API at ${API_BASE_URL}. ${summaryError || comparisonError || retrievalError}`
              : [summaryError, comparisonError, retrievalError].filter(Boolean).length
                ? `${[summaryError, comparisonError, retrievalError].filter(Boolean).join(' ')} Other sections may still be available.`
                : 'Metrics come from saved offline evaluation artifacts. Unavailable metrics are shown as “—”, never as zero.'}
        </span>
        {allFailed && <button className="text-button" onClick={() => setReloadKey((key) => key + 1)}><RefreshCcw size={14} /> Retry</button>}
      </div>

      {loading && <div className="empty-state"><RefreshCcw size={22} /><strong>Loading evaluation data…</strong><span>Reading saved artifacts through the API.</span></div>}

      {!loading && experiments.length > 0 && (
        <section className="evaluation-methods">
          <div className="evaluation-section-heading">
            <div><h2>Available experiment metrics</h2><p>Each experiment keeps its own dataset and metric definitions.</p></div>
            <span className="sample-pill">{summary.available_experiment_count} of {summary.experiment_count} available</span>
          </div>
          {summaryError && <div className="run-message" role="alert">{summaryError}</div>}
          <div className="method-grid">
            {experiments.map((experiment) => {
              const rows = metricRows(experiment);
              return (
                <section className="method-card" key={experiment.name}>
                  <div className="method-card-heading">
                    <div>
                      <h3>{EXPERIMENT_LABELS[experiment.name] || humanize(experiment.name)}</h3>
                      <p>{experiment.dataset?.path || experiment.dataset?.dataset_id || 'Dataset metadata unavailable'}</p>
                    </div>
                    <StatusBadge status={experiment.available ? 'Available' : humanize(experiment.status)} />
                  </div>
                  {experiment.available ? (
                    <div className="method-bars">
                      {rows.map(([label, value]) => (
                        <div className="method-bar-row" key={label}>
                          <div className="method-bar-label"><span>{label}</span><strong>{formatMetric(value, 4)}</strong></div>
                          {typeof value === 'number' && value >= 0 && value <= 1 && (
                            <div className="method-track"><span className="method-bar method-bar--blue" style={{ width: `${value * 100}%` }} /></div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="method-unavailable">{experiment.note || experiment.reason || 'No metric values are available.'}</p>
                  )}
                  <span className="method-footnote">{experiment.source || 'No saved artifact'} · generated {formatDateTime(experiment.generated_at)}</span>
                </section>
              );
            })}
          </div>
        </section>
      )}

      {!loading && (
        <section className="evaluation-methods">
          <div className="evaluation-section-heading">
            <div><h2>Ranking method comparison</h2><p>Bias-aware-only, ECS-only, and RRF-combined rankings, reported per objective.</p></div>
            <span className="sample-pill">{comparison ? humanize(comparison.status) : 'Unavailable'}</span>
          </div>
          {comparisonError && <div className="run-message" role="alert">{comparisonError}</div>}
          {comparison && (
            <p className="state-note">
              Dataset: {comparison.dataset?.dataset_id || 'unavailable'} · K values: {comparison.k_values?.join(', ') || '—'} · RRF k: {formatMetric(comparison.rrf_k)}
            </p>
          )}
          <div className="method-grid">
            {orderedMethods.length > 0 ? orderedMethods.map(([key, method]) => (
              <section className="method-card" key={key}>
                <div className="method-card-heading">
                  <div><h3>{method.display_name || humanize(key)}</h3><p>{method.description || 'Ranking method.'}</p></div>
                  <StatusBadge status={method.available ? 'Computed' : humanize(method.status)} />
                </div>
                {method.available ? (
                  <>
                    {method.metrics && <MetricRows metrics={method.metrics} />}
                    {method.metrics_by_objective && Object.entries(method.metrics_by_objective).map(([objective, metrics]) => (
                      <details className="metric-details" key={objective}>
                        <summary>{objective} metrics</summary>
                        <MetricRows metrics={metrics} />
                      </details>
                    ))}
                  </>
                ) : (
                  <p className="method-unavailable">No metric values are reported for this method ({humanize(method.status)}).</p>
                )}
                <span className="method-footnote">{comparison?.source || 'No saved comparison artifact'}</span>
              </section>
            )) : (
              <section className="method-card">
                <div className="method-card-heading"><div><h3>No comparison artifact</h3><p>No saved prioritization comparison was found.</p></div><StatusBadge status="Unavailable" /></div>
                <p className="method-unavailable">{comparison?.reason || 'Run the offline claim-ranking evaluation to create a comparison report.'}</p>
              </section>
            )}
          </div>
          {comparison && !comparison.available && comparison.reason && <p className="state-note">{comparison.reason}</p>}
        </section>
      )}

      {!loading && (
        <div className="evaluation-lower-grid">
          <section className="evaluation-panel retrieval-panel">
            <div className="evaluation-panel-heading">
              <div><h2>Evidence retrieval</h2><p>Metrics saved by the retrieval evaluation, if one exists.</p></div>
              <div className="panel-icon"><Search size={17} /></div>
            </div>
            {retrievalError && <div className="run-message" role="alert">{retrievalError}</div>}
            <div className="retrieval-metrics">
              {retrievalMetrics.length > 0 ? retrievalMetrics.map(([label, value]) => (
                <div key={label}><span>{label}</span><strong>{formatMetric(value, 4)}</strong><small>{retrieval?.dataset?.name || 'saved retrieval artifact'}</small></div>
              )) : expectedRetrievalMetrics.map((label) => (
                <div key={label}><span>{label}</span><strong>—</strong><small>unavailable</small></div>
              ))}
            </div>
            <p className="state-note">{retrieval?.note || (retrieval?.available ? `Source: ${retrieval.source}` : 'No retrieval evaluation artifact is available.')}</p>
          </section>
          <section className="evaluation-panel responsible-panel">
            <div className="evaluation-panel-heading">
              <div><h2>Report model quality responsibly</h2><p>Recommended reporting checklist.</p></div>
              <div className="panel-icon panel-icon--green"><CheckCircle2 size={17} /></div>
            </div>
            <ul className="quality-checklist">
              <li><CheckCircle2 size={15} /> Precision / recall / Macro F1 per objective</li>
              <li><CheckCircle2 size={15} /> Ranking metrics (nDCG@k, Precision@k)</li>
              <li><CheckCircle2 size={15} /> Retrieval coverage and source diversity</li>
              <li><CheckCircle2 size={15} /> Latency, errors, and human-review agreement</li>
            </ul>
          </section>
        </div>
      )}

      <footer className="page-footer evaluation-footer"><span className="footer-dot" /> Only values present in saved evaluation artifacts are displayed. Pending or missing metrics are unavailable, not zero.</footer>
    </div>
  );
}
