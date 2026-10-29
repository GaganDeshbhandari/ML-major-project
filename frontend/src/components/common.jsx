import React from 'react';
import { Check, FileSpreadsheet, X, UploadCloud } from 'lucide-react';
import { useState } from 'react';

export function PriorityBadge({ priority }) {
  const label = String(priority || 'Unknown');
  return <span className={`priority-badge priority-${label.toLowerCase()}`}><span />{label}</span>;
}

export function StatusBadge({ status }) {
  const label = String(status || 'Unknown');
  return <span className={`status-badge status-${label.toLowerCase().replaceAll(' ', '-')}`}>{label}</span>;
}

export function MetricCard({ label, value, detail }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

export function EvidenceInvestigationCard({ type, title, date, summary, relevance, url, source }) {
  return (
    <article className={`investigation-evidence-card evidence-type-${String(type || 'evidence').toLowerCase()}`}>
      <div className="evidence-card-topline">
        <span className="evidence-type-badge"><span />{type || 'Evidence'}</span>
        <span>{source || 'Stored evidence'}</span>
      </div>
      <h3>{url ? <a href={url} target="_blank" rel="noreferrer">{title}</a> : title}</h3>
      {date && <div className="evidence-date">{date}</div>}
      <p>{summary || 'No summary text was stored for this evidence item.'}</p>
      {relevance !== null && relevance !== undefined && (
        <div className="relevance-row"><span>Stance confidence</span><strong>{formatMetric(relevance, 2)}</strong></div>
      )}
    </article>
  );
}

export function UploadCard({ title, hint, file, onFileChange, validation }) {
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
    <div className={`upload-card ${isDragging ? 'upload-card--dragging' : ''} ${file ? 'upload-card--ready' : ''}`}
      onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => { event.preventDefault(); setIsDragging(false); acceptFile(event.dataTransfer.files[0]); }}>
      <div className="upload-card-header">
        <div className="upload-file-icon"><FileSpreadsheet size={19} /></div>
        <div><h3>{title}</h3><p>{hint}</p></div>
      </div>
      {file ? (
        <div className="selected-file">
          <div className="selected-file-meta"><Check size={16} /><span>{file.name}</span><small>{formatFileSize(file.size)}</small></div>
          <button aria-label={`Remove ${file.name}`} onClick={() => onFileChange(null)}><X size={16} /></button>
        </div>
      ) : (
        <label className="drop-zone" htmlFor={inputId}>
          <UploadCloud size={21} /><span><strong>Choose a CSV</strong> or drag it here</span><small>Maximum file size: 25 MB</small>
          <input id={inputId} type="file" accept=".csv,text/csv" onChange={(event) => acceptFile(event.target.files[0])} />
        </label>
      )}
      {validation && <ValidationStatus validation={validation} />}
      {error && <div className="upload-error" role="alert">{error}</div>}
    </div>
  );
}

function ValidationStatus({ validation }) {
  const { status, validation: result, message } = validation;
  const errors = result?.errors || [];
  const warnings = result?.warnings || [];
  if (status === 'checking') {
    return <div className="upload-validation upload-validation--checking" role="status">Validating CSV with the API…</div>;
  }
  if (status === 'error') {
    return <div className="upload-validation upload-validation--error" role="alert">{message || 'Validation failed.'}</div>;
  }
  return (
    <div className={`upload-validation upload-validation--${status}`} role="status">
      <strong>
        {status === 'valid' ? 'Valid CSV' : 'CSV needs fixes'}
        {result ? ` · ${result.row_count} row${result.row_count === 1 ? '' : 's'}` : ''}
      </strong>
      {errors.length > 0 && <ul>{errors.map((item) => <li key={item}>{item}</li>)}</ul>}
      {warnings.length > 0 && <ul className="upload-warning-list">{warnings.map((item) => <li key={item}>{item}</li>)}</ul>}
    </div>
  );
}

export function Step({ number, title, description }) {
  return <div className="step"><div className="step-number">{number}</div><div><h3>{title}</h3><p>{description}</p></div></div>;
}

export function SummaryRow({ label, value, ready }) {
  return <div className="summary-row"><span>{label}</span><span className={ready ? 'summary-ready' : ''}>{ready && <Check size={13} />}{value}</span></div>;
}

export function formatFileSize(bytes) {
  if (!bytes) return '0 KB';
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function ecsClass(ecs) {
  if (ecs === null || ecs === undefined) return 'unavailable';
  if (ecs >= 0.7) return 'high';
  if (ecs >= 0.4) return 'medium';
  return 'low';
}

export function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  if (typeof value !== 'number') return String(value);
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function formatDateTime(value) {
  if (!value) return 'Unavailable';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

export function humanize(value) {
  if (!value) return 'Unknown';
  return String(value).replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}
