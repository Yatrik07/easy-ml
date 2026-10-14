// Side-by-side comparison of two runs of the same task.
import { reactive } from '../lib/vue.js';
import { getJSON } from '../lib/api.js';
import { navigate, toastError } from './ui.js';

export const compare = reactive({
  loading: false,
  error: '',
  a: null, // { detail, config } of the baseline (older or parent) run
  b: null,
});

async function loadRun(id) {
  const [detail, config] = await Promise.all([
    getJSON('/runs/' + id, 'Could not load run #' + id),
    getJSON('/runs/' + id + '/config', 'Could not load the settings of run #' + id),
  ]);
  return { detail, config };
}

export async function openCompare(firstId, secondId) {
  Object.assign(compare, { loading: true, error: '', a: null, b: null });
  navigate('compare');
  try {
    let [a, b] = await Promise.all([loadRun(firstId), loadRun(secondId)]);
    // The parent (or else the older run) is the baseline.
    const bIsParent = a.detail.parent_run_id === b.detail.id;
    if (bIsParent || (a.detail.parent_run_id !== b.detail.id && b.detail.parent_run_id !== a.detail.id && b.detail.id < a.detail.id)) {
      [a, b] = [b, a];
    }
    compare.a = a;
    compare.b = b;
  } catch (e) {
    compare.error = e.message;
    toastError(e.message);
  } finally {
    compare.loading = false;
  }
}

export function swapCompare() {
  [compare.a, compare.b] = [compare.b, compare.a];
}
