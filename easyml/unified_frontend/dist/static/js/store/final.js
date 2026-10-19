// "Train final model": the chosen models of an evaluated run trained again on all its rows (POST /runs/<id>/final).
import { reactive } from '../lib/vue.js';
import { getJSON } from '../lib/api.js';
import { startFinal } from './training.js';
import { toastError } from './ui.js';

export const finalModel = reactive({
  open: false,
  loading: false,
  run: null,     // the evaluated run: {id, run_name, task, models: [{key, label, score}], metric, metrics}
  chosen: [],    // model class names to train
});

export async function openFinalDialog(runId) {
  Object.assign(finalModel, { open: true, loading: true, run: null, chosen: [] });
  try {
    const run = await getJSON('/runs/' + runId, 'Could not load run #' + runId);
    finalModel.run = run;
    finalModel.chosen = run.best_model ? [run.best_model] : (run.models.length ? [run.models[0].key] : []);
  } catch (e) {
    finalModel.open = false;
    toastError(e.message);
  } finally {
    finalModel.loading = false;
  }
}

export function closeFinalDialog() {
  finalModel.open = false;
}

export function toggleFinalModel(key) {
  const i = finalModel.chosen.indexOf(key);
  if (i >= 0) finalModel.chosen.splice(i, 1);
  else finalModel.chosen.push(key);
}

export function submitFinal() {
  const run = finalModel.run;
  if (!run || !finalModel.chosen.length) return;
  finalModel.open = false;
  return startFinal(run.id, { models: [...finalModel.chosen] }, run.task);
}
