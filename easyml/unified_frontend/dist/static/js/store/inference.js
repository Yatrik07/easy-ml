// Predictions with a saved run: manual input, batch CSV, SHAP explanations and export.
import { reactive } from '../lib/vue.js';
import { getJSON, postForBlob, postForm, postJSON, saveBlob } from '../lib/api.js';
import { naturalSort } from '../lib/format.js';
import { navigate, toastError, toastSuccess } from './ui.js';

export const inference = reactive({
  runId: null,
  task: '',
  runName: '',
  datasetName: '',
  models: [],
  bestModel: null,
  metric: null,
  modelScores: {},
  columns: [],
  columnDetails: {},
  exampleRow: {}, // a real row from the training data, for the API examples
  modelName: '',
  mode: 'manual', // manual | batch
  values: {},
  batchFile: null,
  text: '',
  image: '',
  imagePreview: '',
  result: null,
  predicting: false,
  shapPlot: null,
  explanation: null, // {scale, base_value, output_value, features, other, grouped}
  explaining: false,
  exporting: false,
  sample: null, // a real row put in the form: {actual, row, source, form}
  sampling: false,
  cleaning: null, // {name, steps, automatic}: the data-prep recipe of a run trained on cleaned data
  dataCheck: null, // the last CSV file compared with the training data (POST /runs/<id>/data-check)
  checking: false,
  // Runs with several columns to predict: {columns, weights, classes, scores, combined}; null otherwise.
  targets: null,
  explainTarget: '', // which column "Explain this prediction" is about
});

// The columns a run predicts, when it predicts several (else an empty list).
export function targetList() {
  return (inference.targets && inference.targets.columns) || [];
}

export function columnChoices(column) {
  const details = inference.columnDetails[column];
  return details && details.type === 'categorical' ? naturalSort(details.values || []) : [];
}

// How the form asks for a column: a list of choices, a number, or free text (dates, long text).
export function inputKind(column) {
  const details = inference.columnDetails[column];
  if (details && details.type === 'categorical' && (details.values || []).length) return 'choice';
  if (details && details.type === 'text') return 'text';
  return 'number';
}

export async function openInference(runId) {
  try {
    const data = await getJSON('/get_run_data?run_id=' + runId, 'Could not load run #' + runId);
    Object.assign(inference, {
      runId,
      task: data.task,
      runName: data.run_name || '',
      datasetName: data.dataset_name || '',
      models: data.models || [],
      bestModel: data.best_model,
      metric: data.metric,
      modelScores: data.model_scores || {},
      columns: (data.columns || []).filter(c => c !== data.target_column),
      columnDetails: data.column_details || {},
      exampleRow: data.example_row || {},
      modelName:(data.models || [])[0] || '',
      mode: 'manual',
      values: {},
      batchFile: null,
      text: '',
      image: '',
      imagePreview: '',
      result: null,
      shapPlot: null,
      explanation: null,
      sample: null,
      cleaning: data.cleaning || null,
      dataCheck: null,
      targets: data.targets || null,
      explainTarget: data.targets ? data.targets.columns[0] : '',
    });
    for (const col of inference.columns) inference.values[col] = '';
    navigate('inference');
  } catch (e) {
    toastError(e.message);
  }
}

export async function predict() {
  if (!inference.modelName) return toastError('Choose a model first.');
  inference.predicting = true;
  inference.result = null;
  inference.shapPlot = null;
  inference.explanation = null;
  try {
    const payload = { run_id: inference.runId, model_name: inference.modelName };
    let endpoint = '/inference';
    if (inference.task === 'NLPTextClassification') {
      if (!inference.text.trim()) throw new Error('Enter some text to classify.');
      payload.data = inference.text;
      endpoint = '/nlp/inference';
    } else if (inference.task === 'ImageClassification') {
      if (!inference.image) throw new Error('Choose an image first.');
      payload.data = inference.image;
      endpoint = '/cv/inference';
    } else {
      payload.data = { ...inference.values };
    }
    const result = await postJSON(endpoint, payload, 'Prediction failed');
    inference.result = Array.isArray(result) ? result[0] : result;
  } catch (e) {
    toastError(e.message);
  } finally {
    inference.predicting = false;
  }
}

export async function predictBatch() {
  if (!inference.batchFile) return toastError('Choose a CSV file first.');
  inference.predicting = true;
  try {
    const form = new FormData();
    form.append('run_id', inference.runId);
    form.append('model_name', inference.modelName);
    form.append('file', inference.batchFile);
    const blob = await postForBlob('/inference_batch', form, 'Batch prediction failed');
    saveBlob(blob, 'predictions_run_' + inference.runId + '.csv');
    toastSuccess('Predictions downloaded.');
    checkBatch(inference.batchFile);
  } catch (e) {
    toastError(e.message);
  } finally {
    inference.predicting = false;
  }
}

// Compare the predicted file with the training data. Only informs, so a failure is not shown as an error.
export async function checkBatch(file) {
  if (!file) return;
  inference.checking = true;
  inference.dataCheck = null;
  try {
    const form = new FormData();
    form.append('file', file);
    inference.dataCheck = await postForm('/runs/' + inference.runId + '/data-check', form, 'Data check failed');
  } catch (e) {
    inference.dataCheck = null;
  } finally {
    inference.checking = false;
  }
}

export async function explain() {
  inference.explaining = true;
  try {
    const body = { run_id: inference.runId, model_name: inference.modelName, data: { ...inference.values } };
    if (inference.targets) body.target = inference.explainTarget;
    const result = await postJSON('/explain', body, 'Could not explain this prediction');
    inference.shapPlot = result.plot_b64;
    inference.explanation = result.explanation || null;
  } catch (e) {
    toastError(e.message);
  } finally {
    inference.explaining = false;
  }
}

// True while no input field has a value: predicting then only shows what the model does with missing values.
export function allInputsEmpty() {
  return inference.columns.every(col => {
    const value = inference.values[col];
    return value === '' || value === null || value === undefined;
  });
}

// What is in the form now, to tell whether a sampled row was edited afterwards.
function formState() {
  return inference.task === 'NLPTextClassification' ? inference.text : JSON.stringify(inference.columns.map(c => String(inference.values[c])));
}

// The form value for a value from the data: the matching dropdown option for categories, '' for missing.
function formValue(column, value) {
  if (value === null || value === undefined) return '';
  const choices = columnChoices(column);
  if (!choices.length) return value;
  const match = choices.find(c => String(c) === String(value)
    || (c !== '' && !Number.isNaN(Number(c)) && Number(c) === Number(value)));
  return match === undefined ? '' : match;
}

// Fill the form with a random real row of the run's data (its test set when available) and predict it.
export async function tryRealRow() {
  inference.sampling = true;
  try {
    const sample = await getJSON('/runs/' + inference.runId + '/sample', 'Could not take a row from the data');
    if (inference.task === 'NLPTextClassification') {
      inference.text = sample.text || '';
    } else {
      inference.mode = 'manual';
      for (const col of inference.columns) inference.values[col] = formValue(col, (sample.values || {})[col]);
    }
    inference.sample = { actual: sample.actual, row: sample.row, source: sample.source, form: formState() };
    await predict();
  } catch (e) {
    toastError(e.message);
  } finally {
    inference.sampling = false;
  }
}

// The sampled row's real answer, while the form still holds that row (null otherwise). With several columns to
// predict, ``actual`` is {column: value}.
export function sampleInForm() {
  const sample = inference.sample;
  if (!sample || sample.actual === null || sample.actual === undefined) return null;
  return sample.form === formState() ? sample : null;
}

export async function exportModel() {
  inference.exporting = true;
  try {
    const blob = await postForBlob('/export', { run_id: inference.runId, model_name: inference.modelName }, 'Export failed');
    saveBlob(blob, 'deploy_' + inference.modelName + '_' + inference.runId + '.zip');
    toastSuccess('API package downloaded. Unzip it and follow README.md inside.');
  } catch (e) {
    toastError(e.message);
  } finally {
    inference.exporting = false;
  }
}

export function setImage(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = event => {
    inference.image = event.target.result;
    inference.imagePreview = event.target.result;
  };
  reader.readAsDataURL(file);
}
