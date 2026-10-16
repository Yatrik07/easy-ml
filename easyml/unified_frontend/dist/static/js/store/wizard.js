// The "new run" wizard: chosen task, uploaded dataset, configuration and model selection.
import { reactive, computed, watch } from '../lib/vue.js';
import { getJSON, postJSON, postForm } from '../lib/api.js';
import { session } from './session.js';
import { toastError, toastSuccess } from './ui.js';

function defaultConfig(task = '') {
  return {
    task,
    name: '',
    target_column: '',
    dataset_path: '',
    original_filename: '',
    recipe: null, // data-prep steps that produced the dataset (recorded with the run)
    hyper_parameter_tuning: 0,
    cv_folds: 0, // k-fold cross-validation on the training data (0 = off)
    random_seed: 42, // same data + settings + seed = same models and scores
    models: [],
    imputation_strategy: 'impute',
    imbalance_handling: 'none',
    // Tabular
    categorical_columns: [],
    ignore_columns: [],
    scaling: 'No',
    reduce_features: 'No',
    // Text
    text_column: '',
    lowercase: true,
    remove_stopwords: true,
    lemmatize: false,
    regex_clean: true,
    feature_mode: 'vectorizer',
    vectorizer_type: 'tfidf',
    max_features: 5000,
    embedding_model: 'all-MiniLM-L6-v2',
    // Images
    image_size_input: '224,224',
    batch_size: 32,
    validation_split: 0.2,
    augment: true,
    epochs: 5,
    learning_rate: 0.001,
    early_stopping_patience: 5,
    reduce_lr_patience: 3,
    reduce_lr_factor: 0.5,
  };
}

function defaultLayers() {
  return [
    { type: 'Conv2d', out_channels: 32, kernel_size: 3, padding: 1 },
    { type: 'ReLU' },
    { type: 'MaxPool2d', kernel_size: 2 },
    { type: 'Flatten' },
    { type: 'Linear', out_features: 128 },
    { type: 'ReLU' },
    { type: 'Linear', out_features: 10 },
  ];
}

export const wizard = reactive({
  config: defaultConfig(),
  upload: { fileName: '', data: null, uploading: false },
  cvConfirmed: false,
  cvClassCount: null,
  classDistribution: null,
  classDistributionLoading: false,
  edaLoading: false,
  folderPicker: { open: false, loading: false, folders: [], selected: '' },
  layers: defaultLayers(),
  cnnValidation: null,
  modelsInitialised: false,
});

export const isText = computed(() => wizard.config.task === 'NLPTextClassification');
export const isImage = computed(() => wizard.config.task === 'ImageClassification');
export const isTabular = computed(() => ['MLClassification', 'MLRegression'].includes(wizard.config.task));
export const isClassification = computed(() => ['MLClassification', 'NLPTextClassification'].includes(wizard.config.task));

export const dataReady = computed(() => {
  const c = wizard.config;
  if (!c.task || !c.name.trim()) return false;
  if (isImage.value) return wizard.cvConfirmed && !!c.dataset_path;
  if (!wizard.upload.data || !c.target_column) return false;
  return !isText.value || !!c.text_column;
});

export const featureColumns = computed(() => {
  const data = wizard.upload.data;
  if (!data) return [];
  return data.column_names.filter(c => c !== wizard.config.target_column && c !== wizard.config.text_column);
});

export function resetWizard(task = '') {
  Object.assign(wizard.config, defaultConfig(task));
  wizard.upload = { fileName: '', data: null, uploading: false };
  wizard.cvConfirmed = false;
  wizard.cvClassCount = null;
  wizard.classDistribution = null;
  wizard.layers = defaultLayers();
  wizard.cnnValidation = null;
  wizard.modelsInitialised = false;
  if (session.capabilities && !session.capabilities.features.lemmatize) wizard.config.lemmatize = false;
}

function defaultRunName(fileName) {
  const stem = String(fileName || '').replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 40);
  return stem || 'run';
}

export async function uploadDataset(file) {
  if (!file) return;
  wizard.upload = { fileName: file.name, data: null, uploading: true };
  wizard.config.target_column = '';
  wizard.config.text_column = '';
  const form = new FormData();
  form.append('file', file);
  const endpoint = isText.value ? '/nlp/load_data' : '/load_data';
  try {
    const data = await postForm(endpoint, form, 'Upload failed');
    wizard.upload = { fileName: file.name, data, uploading: false };
    wizard.config.dataset_path = data.saved_path;
    wizard.config.original_filename = data.original_filename || file.name;
    wizard.config.categorical_columns = [];
    wizard.config.ignore_columns = [];
    if (!wizard.config.name) wizard.config.name = defaultRunName(file.name);
    toastSuccess('Loaded ' + file.name + ' (' + data.rows.toLocaleString() + ' rows, ' + data.cols + ' columns)');
  } catch (e) {
    wizard.upload = { fileName: '', data: null, uploading: false };
    toastError(e.message);
  }
}

export function confirmImageFolder() {
  const c = wizard.config;
  if (!c.dataset_path.trim()) {
    toastError('Enter the path of an image folder, or choose one with Browse.');
    return;
  }
  wizard.cvConfirmed = true;
  if (!c.name) {
    const last = c.dataset_path.split(',')[0].trim().split(/[\\/]/).filter(Boolean).pop();
    c.name = defaultRunName(last);
  }
}

// Class balance for classification targets.
watch(() => [wizard.config.target_column, wizard.config.dataset_path], async ([target, path]) => {
  if (!target || !path || !isClassification.value || isImage.value) {
    wizard.classDistribution = null;
    return;
  }
  wizard.classDistributionLoading = true;
  try {
    wizard.classDistribution = await postJSON('/class_distribution', { dataset_path: path, target_column: target });
  } catch (e) {
    wizard.classDistribution = null;
  } finally {
    wizard.classDistributionLoading = false;
  }
});

export async function openFolderPicker() {
  const picker = wizard.folderPicker;
  picker.open = true;
  picker.loading = true;
  picker.selected = wizard.config.dataset_path || '';
  try {
    picker.folders = (await getJSON('/cv/list_datasets', 'Could not list folders')).folders || [];
  } catch (e) {
    toastError(e.message);
  } finally {
    picker.loading = false;
  }
}

export function chooseFolder() {
  const picker = wizard.folderPicker;
  const folder = picker.folders.find(f => f.path === picker.selected);
  wizard.config.dataset_path = picker.selected;
  wizard.cvClassCount = folder ? folder.num_classes : null;
  wizard.cvConfirmed = false;
  picker.open = false;
}

// Generate the ydata-profiling report. The tab is opened synchronously (avoids popup blockers)
// and shows a loading page until the report is ready.
export async function openEdaReport() {
  const path = wizard.config.dataset_path;
  if (!path) return;
  const tab = window.open('', '_blank');
  if (tab) {
    tab.document.write('<!doctype html><title>Generating report…</title><body style="font-family:system-ui,sans-serif;'
      + 'display:grid;place-items:center;height:90vh;color:#52525b"><div><h2>Generating EDA report…</h2>'
      + '<p>This can take up to a minute for large datasets.</p></div></body>');
  }
  wizard.edaLoading = true;
  try {
    const res = await fetch('/generate_eda?path=' + encodeURIComponent(path));
    if (!res.ok) {
      let message = 'Could not generate the report';
      try { message = (await res.json()).error || message; } catch (e) { /* keep default */ }
      throw new Error(message);
    }
    const html = await res.text();
    if (tab) {
      tab.document.open();
      tab.document.write(html);
      tab.document.close();
    }
  } catch (e) {
    if (tab) tab.close();
    toastError(e.message);
  } finally {
    wizard.edaLoading = false;
  }
}

// Sensible starting selection the first time the Models step opens.
const RECOMMENDED = {
  MLClassification: ['logistic_regression', 'random_forest', 'light_gbm'],
  MLRegression: ['linear_regression', 'random_forest', 'light_gbm'],
  NLPTextClassification: ['logistic_regression', 'complement_nb', 'svm'],
  ImageClassification: ['SimpleCNN'],
};

export function availableModels() {
  return session.models[wizard.config.task] || [];
}

export function initialiseModelSelection() {
  if (wizard.modelsInitialised) return;
  const available = availableModels();
  const picks = (RECOMMENDED[wizard.config.task] || []).filter(m => available.includes(m));
  if (!wizard.config.models.length) wizard.config.models = picks;
  wizard.modelsInitialised = true;
}

// ---- Custom CNN builder ----
const LAYER_DEFAULTS = {
  Conv2d: { type: 'Conv2d', out_channels: 32, kernel_size: 3, padding: 1 },
  ReLU: { type: 'ReLU' },
  MaxPool2d: { type: 'MaxPool2d', kernel_size: 2 },
  Dropout: { type: 'Dropout', p: 0.5 },
  Flatten: { type: 'Flatten' },
  Linear: { type: 'Linear', out_features: 128 },
};
export const LAYER_TYPES = Object.keys(LAYER_DEFAULTS);

export function addLayer(type) {
  wizard.layers.push({ ...LAYER_DEFAULTS[type] });
  wizard.cnnValidation = null;
}
export function removeLayer(index) {
  wizard.layers.splice(index, 1);
  wizard.cnnValidation = null;
}
export function moveLayer(index, delta) {
  const target = index + delta;
  if (target < 0 || target >= wizard.layers.length) return;
  const layers = wizard.layers;
  [layers[index], layers[target]] = [layers[target], layers[index]];
  wizard.cnnValidation = null;
}

export function imageSize() {
  return wizard.config.image_size_input.split(',').map(Number);
}

export async function validateCnn() {
  try {
    wizard.cnnValidation = await postJSON('/cv/validate_model', {
      layers: wizard.layers,
      input_size: imageSize(),
      num_classes: wizard.cvClassCount || 2,
    }, 'Validation failed');
  } catch (e) {
    wizard.cnnValidation = { valid: false, error: e.message };
  }
}
