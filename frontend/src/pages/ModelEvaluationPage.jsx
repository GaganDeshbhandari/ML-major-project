import React, { useEffect, useState } from 'react';
import { Info, RefreshCcw } from 'lucide-react';
import { StatusBadge, formatMetric, humanize } from '../components/common';
import {
  API_BASE_URL,
  getEvaluationSummary,
} from '../api/client';

const PRIMARY_EXPERIMENTS = [
  'factllama_fact_verification',
  'nli_stance_classification',
];

const EXPERIMENT_LABELS = {
  factllama_fact_verification: 'FactLLaMA fact verification',
  nli_stance_classification: 'NLI stance classification',
};

function experimentRows(experiment) {
  const metrics = experiment?.metrics || {};
  if (experiment?.name === 'factllama_fact_verification') {
    return [
      ['Accuracy', metrics.accuracy],
      ['Macro F1', metrics.macro_f1],
      ['Scored examples', experiment.counts?.scored],
    ];
  }
  return [
    ['Accuracy', metrics.accuracy],
    ['Macro F1', metrics.macro_f1],
    ['Evaluated examples', metrics.evaluated_examples],
  ];
}

function MetricValues({ metrics }) {
  const rows = Object.entries(metrics || {}).filter(([, value]) => value === null || typeof value !== 'object');
  if (!rows.length) return <p className="method-unavailable">No metric values reported.</p>;
  return (
    <div className="method-bars">
      {rows.map(([label, value]) => (
        <div className="method-bar-row" key={label}>
          <div className="method-bar-label"><span>{label}</span><strong>{formatMetric(value, 4)}</strong></div>
        </div>
      ))}
    </div>
  );
}

export default function ModelEvaluationPage() {
  const [state, setState] = useState({
    loading: true,
    summary: null,
    summaryError: '',
  });
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState((previous) => ({ ...previous, loading: true }));
    getEvaluationSummary().then((summary) => {
      if (cancelled) return;
      setState({
        loading: false,
        summary,
        summaryError: '',
      });
    }).catch((error) => {
      if (cancelled) return;
      setState({
        loading: false,
        summary: null,
        summaryError: error?.message || 'Request failed.',
      });
    });
    return () => { cancelled = true; };
  }, [reloadKey]);

  const { loading, summary, summaryError } = state;
  const allFailed = !loading && !summary;
  const experiments = (summary?.experiments || []).filter((experiment) => PRIMARY_EXPERIMENTS.includes(experiment.name));

  return (
    <div className="evaluation-page">
      <header className="page-header evaluation-header">
        <div>
          <div className="eyebrow">04 · MODEL EVALUATION</div>
          <h1>Model evaluation</h1>
          <p>Read saved evaluation results through the API. No training or inference runs on this page.</p>
        </div>
        <div className="header-status"><span className="status-dot" /> {loading ? 'Loading' : summary ? humanize(summary.status) : 'Unavailable'}</div>
      </header>
      <div className={`evaluation-notice ${allFailed ? 'evaluation-notice--error' : ''}`}>
        <Info size={15} />
        <span>
          {loading
            ? `Loading saved metrics from ${API_BASE_URL}…`
            : allFailed
              ? `Could not reach the evaluation API at ${API_BASE_URL}. ${summaryError}`
                : 'Saved evaluation results only. Unavailable metrics are shown as “—”, never as zero.'}
        </span>
        {allFailed && <button className="text-button" onClick={() => setReloadKey((key) => key + 1)}><RefreshCcw size={14} /> Retry</button>}
      </div>

      {loading && <div className="empty-state"><RefreshCcw size={22} /><strong>Loading evaluation data…</strong><span>Reading saved artifacts through the API.</span></div>}

      {!loading && (
        <section className="evaluation-methods">
          <div className="evaluation-section-heading">
            <div><h2>Saved model results</h2><p>FactLLaMA and NLI metrics from their saved evaluation artifacts.</p></div>
          </div>
          {experiments.length > 0 ? (
            <div className="method-grid">
              {experiments.map((experiment) => (
                <section className="method-card" key={experiment.name}>
                  <div className="method-card-heading">
                    <div><h3>{EXPERIMENT_LABELS[experiment.name] || humanize(experiment.name)}</h3></div>
                    <StatusBadge status={experiment.available ? 'Available' : humanize(experiment.status)} />
                  </div>
                  {experiment.available ? (
                    <MetricValues metrics={Object.fromEntries(experimentRows(experiment))} />
                  ) : (
                    <p className="method-unavailable">{experiment.note || 'No saved metric values are available.'}</p>
                  )}
                </section>
              ))}
            </div>
          ) : (
            <p className="method-unavailable">No saved FactLLaMA or NLI results are available.</p>
          )}
        </section>
      )}

      <footer className="page-footer evaluation-footer"><span className="footer-dot" /> FactLLaMA and NLI stance metrics are read from saved evaluation artifacts.</footer>
    </div>
  );
}
