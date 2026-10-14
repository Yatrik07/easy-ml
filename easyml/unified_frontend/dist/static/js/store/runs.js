// Saved runs (Run History, Home "recent runs", the history detail drawer).
import { reactive } from '../lib/vue.js';
import { deleteJSON, getJSON } from '../lib/api.js';
import { toastError, toastSuccess } from './ui.js';

export const runs = reactive({
  list: [],
  loaded: false,
  loading: false,
  detail: null, // full detail of the run open in the drawer
  detailLoading: false,
});

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

export async function deleteRun(id) {
  try {
    await deleteJSON('/runs', { run_id: id }, 'Could not delete run #' + id);
    runs.list = runs.list.filter(r => r.id !== id);
    if (runs.detail && runs.detail.id === id) runs.detail = null;
    toastSuccess('Run #' + id + ' deleted.');
  } catch (e) {
    toastError(e.message);
  }
}
