// The "Retrain on new data" dialog: loads what a run expects, uploads the new data and starts the job.
import { reactive } from '../lib/vue.js';
import { getJSON, postForm } from '../lib/api.js';
import { toastError } from './ui.js';
import { isRunning, startRetrain } from './training.js';
import { closeRunDetail } from './runs.js';

export const retrain = reactive({
  open: false,
  runId: null,
  spec: null,
  loading: false,
  error: '',
  upload: { fileName: '', uploading: false, columns: null, savedPath: '', originalFilename: '', rows: 0 },
  folders: [],
  folder: '',
  name: '',
  recipes: [],      // saved data-prep recipes [{id, name, steps}]
  recipeId: '',     // '' = automatic (the run's own recipe, if any), 'none' = use the file as it is, or a recipe id
});

function emptyUpload() {
  return { fileName: '', uploading: false, columns: null, savedPath: '', originalFilename: '', rows: 0 };
}

export async function openRetrain(runId) {
  Object.assign(retrain, {
    open: true, runId, spec: null, loading: true, error: '', upload: emptyUpload(), folders: [], folder: '', name: '',
    recipes: [], recipeId: '',
  });
  try {
    retrain.spec = await getJSON('/runs/' + runId + '/config', 'Could not load run #' + runId);
    retrain.name = retrain.spec.retrain.default_name || '';
    if (retrain.spec.retrain.input === 'image_folder') {
      retrain.folders = (await getJSON('/cv/list_datasets', 'Could not list image folders')).folders || [];
    } else {
      // Optional: without saved recipes the dialog simply offers none.
      retrain.recipes = (await getJSON('/prep/recipes', 'Could not load recipes').catch(() => ({ recipes: [] }))).recipes || [];
    }
  } catch (e) {
    retrain.error = e.message;
  } finally {
    retrain.loading = false;
  }
}

export function closeRetrain() {
  retrain.open = false;
}

export async function uploadRetrainFile(file) {
  if (!file || !retrain.spec) return;
  retrain.upload = Object.assign(emptyUpload(), { fileName: file.name, uploading: true });
  const form = new FormData();
  form.append('file', file);
  try {
    const data = await postForm(retrain.spec.retrain.upload_endpoint, form, 'Upload failed');
    retrain.upload = {
      fileName: file.name, uploading: false, columns: data.column_names || [], savedPath: data.saved_path,
      originalFilename: data.original_filename || file.name, rows: data.rows || 0,
    };
  } catch (e) {
    retrain.upload = emptyUpload();
    toastError(e.message);
  }
}

// The recipe that will clean the new file before training, or null when it is used as it is.
export function cleaningRecipe() {
  const spec = retrain.spec && retrain.spec.retrain;
  if (!spec || spec.input === 'image_folder' || retrain.recipeId === 'none') return null;
  if (retrain.recipeId === '') return spec.recipe ? { name: spec.recipe.name, steps: spec.recipe.steps, own: true } : null;
  const saved = retrain.recipes.find(r => String(r.id) === String(retrain.recipeId));
  return saved ? { name: saved.name, steps: saved.steps, own: false } : null;
}

// Columns the run needs that the uploaded file lacks, and columns it has that the run never used.
export function columnCheck() {
  const spec = retrain.spec && retrain.spec.retrain;
  const columns = retrain.upload.columns;
  if (!spec || !columns) return { missing: [], extra: [] };
  const required = spec.required_columns || [];
  return {
    missing: required.filter(c => !columns.includes(c)),
    extra: spec.input === 'table' ? columns.filter(c => !required.includes(c)) : [],
  };
}

export async function submitRetrain() {
  const spec = retrain.spec;
  if (!spec) return;
  if (isRunning()) {
    toastError('Another run is still training. Wait for it to finish or stop it first.');
    return;
  }
  const body = { name: retrain.name };
  if (spec.retrain.input === 'image_folder') {
    body.dataset_path = retrain.folder;
  } else {
    body.dataset_path = retrain.upload.savedPath;
    body.original_filename = retrain.upload.originalFilename;
    if (retrain.recipeId) body.recipe_id = retrain.recipeId;
  }
  retrain.open = false;
  closeRunDetail();
  await startRetrain(retrain.runId, body, spec.task);
}
