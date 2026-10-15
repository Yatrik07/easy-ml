// Starting the new-run wizard, empty or with a dataset that is already uploaded (e.g. from Data prep).
import { confirmAction } from './ui.js';
import { isRunning, resetTraining, training } from './training.js';
import { resetWizard, wizard } from './wizard.js';

// Ask before leaving a training run that is still going; returns false if the user cancels.
async function readyForNewRun() {
  if (isRunning()) {
    const ok = await confirmAction({
      title: 'Training in progress',
      message: 'A model is still training. Starting a new run stops following it here (it keeps running on the server).',
      confirmLabel: 'Start new run',
    });
    if (!ok) return false;
  }
  if (training.status !== 'running') resetTraining();
  return true;
}

export async function beginNewRun(task) {
  if (!(await readyForNewRun())) return false;
  resetWizard(task);
  return true;
}

function runName(fileName) {
  const stem = String(fileName || '').replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 40);
  return stem || 'run';
}

// `data` has the /load_data shape (column_names, rows, cols, data_head, saved_path, original_filename).
// options: { target_column, text_column, name, recipe }
export async function startWizardWithDataset(task, data, options = {}) {
  if (!(await readyForNewRun())) return false;
  resetWizard(task);
  wizard.upload = { fileName: data.original_filename, data, uploading: false };
  Object.assign(wizard.config, {
    dataset_path: data.saved_path,
    original_filename: data.original_filename,
    name: options.name || runName(data.original_filename),
    target_column: options.target_column || '',
    text_column: options.text_column || '',
    recipe: options.recipe || null,
  });
  return true;
}
