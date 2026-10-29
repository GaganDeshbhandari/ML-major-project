const env = import.meta.env || {};
const configuredBase = env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const API_BASE_URL = configuredBase.replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function errorMessage(body, fallback) {
  if (!body) return fallback;
  if (typeof body === 'string') return body;
  if (typeof body.detail === 'string') return body.detail;
  if (body.detail && typeof body.detail === 'object') {
    const detail = body.detail;
    const parts = [];
    if (detail.message) parts.push(detail.message);
    if (Array.isArray(detail.errors)) parts.push(...detail.errors);
    if (detail.detail) parts.push(String(detail.detail));
    if (parts.length) return parts.join(' ');
  }
  if (body.message) return body.message;
  return fallback;
}

async function request(path, { method = 'GET', body, headers, signal } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      body,
      headers,
      signal,
    });
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    throw new ApiError(
      `Cannot reach the API at ${API_BASE_URL}. Check that the backend is running and CORS allows this origin.`,
      0,
      null,
    );
  }

  const contentType = response.headers.get('content-type') || '';
  let payload = null;
  if (contentType.includes('application/json')) {
    payload = await response.json().catch(() => null);
  } else if (response.status !== 204) {
    payload = await response.text().catch(() => null);
  }

  if (!response.ok) {
    const fallback = `Request failed with status ${response.status}.`;
    throw new ApiError(errorMessage(payload, fallback), response.status, payload);
  }
  return payload;
}

function formData(file, datasetType) {
  const data = new FormData();
  data.append('file', file);
  data.append('dataset_type', datasetType);
  return data;
}

function queryString(params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    search.set(key, String(value));
  });
  const text = search.toString();
  return text ? `?${text}` : '';
}

export function validateDataset(file, datasetType) {
  return request('/datasets/validate', {
    method: 'POST',
    body: formData(file, datasetType),
  });
}

export function previewDataset(file, datasetType) {
  return request('/datasets/preview', {
    method: 'POST',
    body: formData(file, datasetType),
  });
}

export function createPrioritizationRun({ claimsFile, perceptionFile, objective }) {
  const data = new FormData();
  data.append('claims_file', claimsFile);
  data.append('perception_file', perceptionFile);
  data.append('objective', objective);
  return request('/prioritization-runs', { method: 'POST', body: data });
}

export function getRunStatus(runId) {
  return request(`/prioritization-runs/${encodeURIComponent(runId)}/status`);
}

export function getRunSummary(runId) {
  return request(`/prioritization-runs/${encodeURIComponent(runId)}/summary`);
}

export function getRunResults(runId) {
  return request(`/prioritization-runs/${encodeURIComponent(runId)}/results`);
}

export function listRunClaims(runId, params = {}) {
  return request(
    `/prioritization-runs/${encodeURIComponent(runId)}/claims${queryString(params)}`,
  );
}

export function getRunClaim(runId, claimId) {
  return request(
    `/prioritization-runs/${encodeURIComponent(runId)}/claims/${encodeURIComponent(claimId)}`,
  );
}

export async function downloadRunExport(runId) {
  let response;
  try {
    response = await fetch(
      `${API_BASE_URL}/prioritization-runs/${encodeURIComponent(runId)}/export`,
    );
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${API_BASE_URL}. Check that the backend is running and CORS allows this origin.`,
      0,
      null,
    );
  }
  if (!response.ok) {
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json')
      ? await response.json().catch(() => null)
      : await response.text().catch(() => null);
    throw new ApiError(
      errorMessage(payload, `Export failed with status ${response.status}.`),
      response.status,
      payload,
    );
  }
  const blob = await response.blob();
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `claimrank_${runId}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(link.href);
}

export function createFactCheck(runId, claimId) {
  return request(
    `/prioritization-runs/${encodeURIComponent(runId)}/claims/${encodeURIComponent(claimId)}/fact-check`,
    { method: 'POST' },
  );
}

export function getFactCheckJob(jobId) {
  return request(`/fact-check-jobs/${encodeURIComponent(jobId)}`);
}

export function getEvaluationSummary() {
  return request('/evaluation/summary');
}

export function getPrioritizationComparison() {
  return request('/evaluation/prioritization-comparison');
}

export function getRetrievalMetrics() {
  return request('/evaluation/retrieval-metrics');
}
