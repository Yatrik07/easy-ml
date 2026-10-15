// Data prep: the loaded dataset, the list of steps, the live preview and saved recipes.
import { reactive } from '../lib/vue.js';
import { deleteJSON, getJSON, postForBlob, postForm, postJSON, putJSON, saveBlob } from '../lib/api.js';
import { confirmAction, navigate, toastError, toastSuccess } from './ui.js';
import { startWizardWithDataset } from './newrun.js';

export const prep = reactive({
  specs: [],        // [{op, label, category, help, fields}]
  categories: [],
  dataset: null,    // {path, fileName, rows, cols}
  uploading: false,
  steps: [],        // [{op, params, enabled}]
  openStep: -1,     // index of the step whose settings are shown
  result: null,     // the last /prep/preview answer
  loading: false,
  error: '',
  recipe: null,     // {id, name} of the loaded / saved recipe
  dirty: false,     // steps changed since the recipe was loaded or saved
  recipes: [],
  busy: '',         // 'download' | 'apply' | '' while a full-file action runs
});

let previewTimer = null;
let previewSeq = 0;

export function specFor(op) {
  return prep.specs.find(s => s.op === op) || null;
}

export async function loadSpecs() {
  if (prep.specs.length) return;
  try {
    const data = await getJSON('/prep/steps', 'Could not load the data-prep steps');
    prep.specs = data.steps;
    prep.categories = data.categories;
  } catch (e) {
    toastError(e.message);
  }
}

function defaultParams(spec) {
  const params = {};
  for (const field of spec.fields) {
    if (field.default !== undefined) params[field.name] = Array.isArray(field.default) ? [...field.default] : field.default;
    else if (field.type === 'columns' || field.type === 'parts') params[field.name] = [];
    else if (field.type === 'mapping') params[field.name] = {};
    else if (field.type === 'bool') params[field.name] = false;
    else params[field.name] = '';
  }
  return params;
}

// Steps as the server expects them (no UI-only fields, no empty optional values).
export function cleanSteps() {
  return prep.steps.map(step => {
    const params = {};
    for (const [key, value] of Object.entries(step.params)) {
      if (value === '' || value === null || value === undefined) continue;
      params[key] = value;
    }
    return { op: step.op, params, enabled: step.enabled };
  });
}

export async function uploadPrepFile(file) {
  if (!file) return;
  prep.uploading = true;
  const form = new FormData();
  form.append('file', file);
  try {
    const data = await postForm('/load_data', form, 'Upload failed');
    useDataset({ path: data.saved_path, fileName: data.original_filename || file.name, rows: data.rows, cols: data.cols });
  } catch (e) {
    toastError(e.message);
  } finally {
    prep.uploading = false;
  }
}

// Start cleaning a dataset that is already uploaded (from this page or the new-run wizard).
export function useDataset(dataset) {
  prep.dataset = dataset;
  prep.result = null;
  prep.error = '';
  refreshPreview(true);
}

export function closeDataset() {
  prep.dataset = null;
  prep.result = null;
  prep.steps = [];
  prep.recipe = null;
  prep.dirty = false;
  prep.openStep = -1;
}

export function refreshPreview(now) {
  clearTimeout(previewTimer);
  if (!prep.dataset) return;
  if (now) runPreview();
  else previewTimer = setTimeout(runPreview, 350);
}

async function runPreview() {
  const seq = ++previewSeq;
  prep.loading = true;
  try {
    const result = await postJSON('/prep/preview', { dataset_path: prep.dataset.path, steps: cleanSteps() },
      'Could not preview the steps');
    if (seq === previewSeq) {
      prep.result = result;
      prep.error = '';
    }
  } catch (e) {
    if (seq === previewSeq) prep.error = e.message;
  } finally {
    if (seq === previewSeq) prep.loading = false;
  }
}

function changed() {
  prep.dirty = true;
  refreshPreview(false);
}

export function addStep(op) {
  const spec = specFor(op);
  if (!spec) return;
  prep.steps.push({ op, params: defaultParams(spec), enabled: true });
  prep.openStep = prep.steps.length - 1;
  // Steps with required settings run once they are filled in; others show their effect right away.
  changed();
}

export function updateParam(index, name, value) {
  prep.steps[index].params[name] = value;
  changed();
}

export function toggleStep(index) {
  prep.steps[index].enabled = !prep.steps[index].enabled;
  changed();
}

export function moveStep(index, delta) {
  const target = index + delta;
  if (target < 0 || target >= prep.steps.length) return;
  const [step] = prep.steps.splice(index, 1);
  prep.steps.splice(target, 0, step);
  if (prep.openStep === index) prep.openStep = target;
  changed();
}

export function removeStep(index) {
  prep.steps.splice(index, 1);
  if (prep.openStep === index) prep.openStep = -1;
  else if (prep.openStep > index) prep.openStep -= 1;
  changed();
}

// Columns a step can choose from: the columns its input has (after the steps before it).
export function columnsBefore(index) {
  const reports = (prep.result && prep.result.steps) || [];
  if (reports[index]) return reports[index].columns_in;
  return (prep.result && prep.result.columns) || [];
}

// --------------------------------------------------------------------------- recipes

export async function loadRecipes() {
  try {
    prep.recipes = (await getJSON('/prep/recipes', 'Could not load recipes')).recipes;
  } catch (e) {
    toastError(e.message);
  }
}

export async function useRecipe(id) {
  try {
    const recipe = await getJSON('/prep/recipes/' + id, 'Could not load the recipe');
    prep.steps = recipe.steps.map(s => ({ op: s.op, params: { ...s.params }, enabled: s.enabled !== false }));
    prep.recipe = { id: recipe.id, name: recipe.name };
    prep.dirty = false;
    prep.openStep = -1;
    refreshPreview(true);
    toastSuccess('Loaded recipe "' + recipe.name + '".');
  } catch (e) {
    toastError(e.message);
  }
}

// The columns of the original file (what the recipe expects as input).
function sourceColumns() {
  const result = prep.result;
  if (!result) return [];
  return result.steps.length ? result.steps[0].columns_in : result.columns;
}

export async function saveRecipe(name, asNew) {
  const body = { name, steps: cleanSteps(), source_columns: sourceColumns() };
  try {
    const saved = prep.recipe && !asNew
      ? await putJSON('/prep/recipes/' + prep.recipe.id, body, 'Could not save the recipe')
      : await postJSON('/prep/recipes', body, 'Could not save the recipe');
    prep.recipe = { id: saved.id, name: saved.name };
    prep.dirty = false;
    toastSuccess('Saved recipe "' + saved.name + '".');
    loadRecipes();
    return true;
  } catch (e) {
    toastError(e.message);
    return false;
  }
}

export async function deleteRecipe(recipe) {
  const ok = await confirmAction({
    title: 'Delete recipe "' + recipe.name + '"?',
    message: 'The recipe is removed. Runs trained with it keep a copy of its steps.',
    confirmLabel: 'Delete recipe',
    danger: true,
  });
  if (!ok) return;
  try {
    await deleteJSON('/prep/recipes/' + recipe.id, {}, 'Could not delete the recipe');
    if (prep.recipe && prep.recipe.id === recipe.id) prep.recipe = null;
    loadRecipes();
  } catch (e) {
    toastError(e.message);
  }
}

// --------------------------------------------------------------------------- full-file actions

export async function downloadCsv() {
  prep.busy = 'download';
  try {
    const blob = await postForBlob('/prep/download', {
      dataset_path: prep.dataset.path, steps: cleanSteps(), original_filename: prep.dataset.fileName,
    }, 'Download failed');
    saveBlob(blob, cleanFileName());
  } catch (e) {
    toastError(e.message);
  } finally {
    prep.busy = '';
  }
}

export function cleanFileName() {
  const stem = String((prep.dataset && prep.dataset.fileName) || 'dataset').replace(/\.(csv|xlsx|xls)$/i, '');
  return stem.replace(/_clean$/, '') + '_clean.csv';
}

// Clean the whole file and open the new-run wizard with the result.
export async function trainWithData(task, options) {
  prep.busy = 'apply';
  try {
    const data = await postJSON('/prep/apply', {
      dataset_path: prep.dataset.path, steps: cleanSteps(), original_filename: prep.dataset.fileName,
    }, 'Could not apply the steps');
    if (data.failed_steps) toastError(data.failed_steps + ' step(s) could not run and were skipped.');
    const recipe = { name: prep.recipe && !prep.dirty ? prep.recipe.name : 'Unsaved steps', steps: cleanSteps() };
    const started = await startWizardWithDataset(task, data, { ...options, recipe });
    if (started) navigate('upload');
  } catch (e) {
    toastError(e.message);
  } finally {
    prep.busy = '';
  }
}
