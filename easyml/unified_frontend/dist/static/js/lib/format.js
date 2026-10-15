// Formatting and metric helpers shared by every view.

export const OVERFIT_GAP = { percent: 0.10, decimal: 0.15 };

export const TASK_ICONS = {
  MLClassification: 'category',
  MLRegression: 'trending_up',
  NLPTextClassification: 'article',
  ImageClassification: 'image',
};

export function metricFor(task) {
  return task === 'MLRegression'
    ? { key: 'r2', name: 'R²', kind: 'decimal' }
    : { key: 'accuracy', name: 'Accuracy', kind: 'percent' };
}

export function formatScore(value, metric) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const kind = metric && metric.kind ? metric.kind : 'percent';
  if (kind === 'decimal') return Number(value).toFixed(4);
  return (Number(value) * 100).toFixed(1) + '%';
}

export function formatPercent(value, digits = 1) {
  if (value === null || value === undefined) return '—';
  return (Number(value) * 100).toFixed(digits) + '%';
}

export function formatNumber(value, maxFraction = 4) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num.toLocaleString(undefined, { maximumFractionDigits: maxFraction });
}

export function formatPrediction(value) {
  if (typeof value === 'number') return formatNumber(value, 4);
  return value === null || value === undefined ? '—' : String(value);
}

// MAPE is a ratio; enormous values (targets near zero) are meaningless, so hide them.
export function formatMape(value) {
  if (value === null || value === undefined) return '—';
  if (value > 10) return '—';
  return (value * 100).toFixed(1) + '%';
}

function readMetric(entry, key) {
  if (!entry) return null;
  const tm = entry.test_metrics || {};
  if (key === 'mae') return tm.mae ?? null;
  if (key === 'mape') return tm.mape ?? null;
  if (key === 'f1') return entry.f1_test ?? tm.f1_weighted ?? null;
  return null;
}

// Rank the models of a run (performance/metrics dict keyed by model) best first.
export function rankPerformance(task, performance, labelFn) {
  const rows = Object.entries(performance || {}).map(([key, entry]) => ({
    key,
    label: labelFn ? labelFn(key) : key,
    test: entry && entry.accuracy_test != null ? entry.accuracy_test : null,
    train: entry && entry.accuracy_train != null ? entry.accuracy_train : null,
    f1: readMetric(entry, 'f1'),
    mae: readMetric(entry, 'mae'),
    mape: readMetric(entry, 'mape'),
    entry,
  }));
  rows.sort((a, b) => {
    if (a.test === null && b.test !== null) return 1;
    if (b.test === null && a.test !== null) return -1;
    if ((b.test || 0) !== (a.test || 0)) return (b.test || 0) - (a.test || 0);
    return (b.f1 || 0) - (a.f1 || 0);
  });
  return rows;
}

export function overfitHint(row, metric) {
  if (row.train == null || row.test == null) return null;
  const gap = row.train - row.test;
  const limit = OVERFIT_GAP[metric.kind] || 0.1;
  if (gap < limit) return null;
  return 'Training score is much higher than the test score (' + formatScore(row.train, metric) + ' vs '
    + formatScore(row.test, metric) + '). The model may be overfitting.';
}

export function classLabels(entry) {
  const tm = (entry && entry.test_metrics) || {};
  if (Array.isArray(tm.confusion_matrix_order)) return tm.confusion_matrix_order.map(String);
  const report = tm.classification_report || {};
  return Object.keys(report).filter(k => !['accuracy', 'macro avg', 'weighted avg', 'micro avg'].includes(k));
}

export function parseTimestamp(ts) {
  if (!ts) return null;
  const clean = String(ts).replace(' ', 'T').replace(/(\d{4}-\d{2}-\d{2}T\d{2})-(\d{2})-(\d{2})/, '$1:$2:$3');
  const date = new Date(clean);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function relativeTime(ts) {
  const date = parseTimestamp(ts);
  if (!date) return ts || '—';
  const seconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (seconds < 45) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return minutes + ' min ago';
  const hours = Math.round(minutes / 60);
  if (hours < 24) return hours + (hours === 1 ? ' hour ago' : ' hours ago');
  const days = Math.round(hours / 24);
  if (days < 7) return days + (days === 1 ? ' day ago' : ' days ago');
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

// A future moment, for "next run": "in 5 min", "in 3 hours", or the date and time.
export function untilTime(ts) {
  const date = parseTimestamp(ts);
  if (!date) return ts || '—';
  const minutes = Math.round((date.getTime() - Date.now()) / 60000);
  if (minutes < 1) return 'due now';
  if (minutes < 60) return 'in ' + minutes + ' min';
  const hours = Math.round(minutes / 60);
  if (hours < 24) return 'in ' + hours + (hours === 1 ? ' hour' : ' hours');
  return date.toLocaleString(undefined, { weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function absoluteTime(ts) {
  const date = parseTimestamp(ts);
  return date ? date.toLocaleString() : (ts || '');
}

export function naturalSort(values) {
  return [...values].sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' }));
}

// Progress messages come from the server with a leading emoji; the UI shows its own status icons.
export function stripLeadingEmoji(text) {
  return String(text || '').replace(/^[^\p{L}\p{N}(\[]+/u, '').trim();
}

export function logKind(entry) {
  const raw = String((entry && entry.message) || '');
  if (entry && entry.kind) return entry.kind;
  if (entry && entry.done && entry.result && entry.result.status === 'error') return 'error';
  if (/^\s*(❌|Error)/u.test(raw)) return 'error';
  if (/^\s*(⚠️|⛔|⏹)/u.test(raw)) return 'warning';
  if (/^\s*✅/u.test(raw) || (entry && entry.done)) return 'success';
  return 'info';
}

export function truncate(text, max) {
  const s = String(text === null || text === undefined ? '' : text);
  return s.length > max ? s.slice(0, max) + '…' : s;
}

export function initials(name) {
  return String(name || '?').split(/\s+/).filter(Boolean).slice(0, 2).map(p => p[0].toUpperCase()).join('');
}
