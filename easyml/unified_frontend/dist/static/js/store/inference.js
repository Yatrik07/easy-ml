// Predictions with a saved run: manual input, batch CSV, SHAP explanations and export.
import { reactive } from '../lib/vue.js';
import { getJSON, postForBlob, postJSON, saveBlob } from '../lib/api.js';
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
  explaining: false,
  exporting: false,
});

export function columnChoices(column) {
  const details = inference.columnDetails[column];
  return details && details.type === 'categorical' ? naturalSort(details.values || []) : [];
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
  } catch (e) {
    toastError(e.message);
  } finally {
    inference.predicting = false;
  }
}

export async function explain() {
  inference.explaining = true;
  try {
    const result = await postJSON('/explain', {
      run_id: inference.runId, model_name: inference.modelName, data: { ...inference.values },
    }, 'Could not explain this prediction');
    inference.shapPlot = result.plot_b64;
  } catch (e) {
    toastError(e.message);
  } finally {
    inference.explaining = false;
  }
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
