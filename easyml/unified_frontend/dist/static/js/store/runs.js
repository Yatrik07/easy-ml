// Saved runs (Run History, Home "recent runs", the history detail drawer).
import { reactive } from '../lib/vue.js';
import { deleteJSON, getJSON, putJSON } from '../lib/api.js';
import { toastError, toastSuccess } from './ui.js';

export const runs = reactive({
  list: [],
  loaded: false,
  loading: false,
  detail: null, // full detail of the run open in the drawer
  detailLoading: false,
  selected: [], // up to two run ids picked for comparison
});

// Two runs of the same task can be compared.
export function canSelect(run) {
  if (runs.selected.includes(run.id)) return true;
  if (runs.selected.length >= 2) return false;
  const first = runs.list.find(r => r.id === runs.selected[0]);
  return !first || first.task === run.task;
}

export function toggleSelected(run) {
  if (runs.selected.includes(run.id)) runs.selected = runs.selected.filter(id => id !== run.id);
  else if (canSelect(run)) runs.selected = runs.selected.concat([run.id]);
}

export function clearSelected() {
  runs.selected = [];
}

export async function loadRuns() {
  runs.loading = true;
  try {
    runs.list = (await getJSON('/runs', 'Could not load runs')).data || [];
    runs.loaded = true;
  } catch (e) {
    toastError(e.message);
  } finally {
    runs.loading = false;
  }
}

export async function openRunDetail(id) {
  runs.detail = { id };
  runs.detailLoading = true;
  try {
    runs.detail = await getJSON('/runs/' + id, 'Could not load run #' + id);
  } catch (e) {
    runs.detail = null;
    toastError(e.message);
  } finally {
    runs.detailLoading = false;
  }
}

export function closeRunDetail() {
  runs.detail = null;
}

// Change a run's name, notes or tags; the list and the open drawer show the result at once.
export async function saveRunMeta(id, patch) {
  try {
    const saved = await putJSON('/runs/' + id + '/meta', patch, 'Could not save the run');
    const fields = { run_name: saved.run_name, tags: saved.tags, notes: saved.notes };
    const row = runs.list.find(r => r.id === id);
    if (row) Object.assign(row, fields);
    if (runs.detail && runs.detail.id === id) Object.assign(runs.detail, fields);
    return true;
  } catch (e) {
    toastError(e.message);
    return false;
  }
}

export async function deleteRun(id) {
  try {
    await deleteJSON('/runs', { run_id: id }, 'Could not delete run #' + id);
    runs.list = runs.list.filter(r => r.id !== id);
    runs.selected = runs.selected.filter(r => r !== id);
    if (runs.detail && runs.detail.id === id) runs.detail = null;
    toastSuccess('Run #' + id + ' deleted.');
  } catch (e) {
    toastError(e.message);
  }
}
