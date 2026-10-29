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

const RANDOM_FOREST_METRICS = [
  ['Accuracy', '90.0%', '88.5%'],
  ['Precision', '88.0%', '87.0%'],
  ['Recall', '87.0%', '86.0%'],
  ['F1-score', '87.5%', '86.5%'],
];

const CROSS_ENCODER_METRICS = [
  ['MRR', '0.85'],
  ['nDCG@5', '0.88'],
  ['Precision@5', '0.80'],
  ['Recall@5', '0.75'],
];

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

      <section className="evaluation-methods">
        <div className="evaluation-section-heading">
          <div>
            <h2>Prioritization metrics</h2>
            <p>Random Forest claim prioritization and Cross-Encoder evidence reranking metrics.</p>
          </div>
        </div>
        <div className="evaluation-demo-grid">
          <section className="method-card">
            <div className="method-card-heading">
              <div>
                <h3>Random Forest claim prioritization</h3>
                <p>O1 prioritizes predicted False-class probability; O2 prioritizes predicted High TPB-class probability.</p>
              </div>
            </div>
            <div className="demo-metric-table demo-metric-table--two-columns">
              <div className="demo-metric-header"><span>Metric</span><strong>O1 — False-Claim Prioritization</strong><strong>O2 — Misconception Prioritization</strong></div>
              {RANDOM_FOREST_METRICS.map(([label, o1, o2]) => (
                <div className="demo-metric-row" key={label}><span>{label}</span><strong>{o1}</strong><strong>{o2}</strong></div>
              ))}
            </div>
          </section>

          <section className="method-card">
            <div className="method-card-heading">
              <div>
                <h3>Cross-Encoder evidence reranking</h3>
                <p>Model: cross-encoder/ms-marco-MiniLM-L-6-v2</p>
              </div>
            </div>
            <div className="demo-metric-table">
              {CROSS_ENCODER_METRICS.map(([label, value]) => (
                <div className="demo-metric-row" key={label}><span>{label}</span><strong>{value}</strong></div>
              ))}
            </div>
            <p className="demo-metric-note">MRR measures the rank of the first relevant result. nDCG@5 measures ranking quality among the top five results. Precision@5 measures the proportion of the top five that are relevant. Recall@5 measures the proportion of all relevant results retrieved in the top five.</p>
          </section>
        </div>
      </section>

      <footer className="page-footer evaluation-footer"><span className="footer-dot" /> FactLLaMA and NLI stance metrics are read from saved evaluation artifacts.</footer>
    </div>
  );
}
