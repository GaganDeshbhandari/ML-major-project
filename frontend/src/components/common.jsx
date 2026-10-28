import { Check, FileSpreadsheet, X, UploadCloud } from 'lucide-react';
import { useState } from 'react';

export function PriorityBadge({ priority }) {
  return <span className={`priority-badge priority-${priority.toLowerCase()}`}><span />{priority}</span>;
}

export function StatusBadge({ status }) {
  return <span className={`status-badge status-${status.toLowerCase().replace(' ', '-')}`}>{status}</span>;
}

export function MetricCard({ label, value, detail }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

export function EvidenceInvestigationCard({ type, title, date, summary, relevance }) {
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

export function UploadCard({ title, hint, file, onFileChange }) {
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
      {error && <div className="upload-error" role="alert">{error}</div>}
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
  if (ecs >= 0.7) return 'high';
  if (ecs >= 0.4) return 'medium';
  return 'low';
}
