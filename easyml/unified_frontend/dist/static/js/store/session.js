// What this installation can do (optional extras, available models) and display labels.
import { reactive } from '../lib/vue.js';
import { getJSON } from '../lib/api.js';

export const session = reactive({
  version: document.documentElement.dataset.version || '',
  capabilities: null,
  models: {},
});

export async function loadSession() {
  try {
    const [capabilities, models] = await Promise.all([
      getJSON('/settings/capabilities'),
      getJSON('/get_models'),
    ]);
    session.capabilities = capabilities;
    session.models = models;
  } catch (e) {
    // The UI still works with defaults; individual pages report their own errors.
  }
}

const FALLBACK_TASK_LABELS = {
  MLClassification: 'Tabular Classification',
  MLRegression: 'Tabular Regression',
  NLPTextClassification: 'Text Classification',
  ImageClassification: 'Image Classification',
};

export function taskLabel(task) {
  const labels = session.capabilities && session.capabilities.labels;
  return (labels && labels.tasks[task]) || FALLBACK_TASK_LABELS[task] || task || '';
}

export function modelLabel(name) {
  const labels = session.capabilities && session.capabilities.labels;
  return (labels && labels.models[name]) || name;
}

export function featureAvailable(name) {
  const caps = session.capabilities;
  if (!caps) return true; // assume available until loaded, so nothing flickers disabled
  if (name === 'boost') return (caps.models.MLClassification || []).includes('xg_boost');
  return !!caps.features[name];
}

export function installHint(name) {
  const hints = session.capabilities ? session.capabilities.install_hints : {};
  return hints[name] || 'pip install "easy-ml-studio[all]"';
}
