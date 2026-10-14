// Result chart data per run (GET /runs/<id>/charts), fetched once and cached.
import { reactive } from '../lib/vue.js';
import { getJSON } from '../lib/api.js';

export const charts = reactive({
  byRun: {},   // run id -> response ({ available, models, ... })
  loading: {}, // run id -> true while loading
  error: {},   // run id -> message
});

export async function loadCharts(runId, force) {
  if (!runId || charts.loading[runId] || (charts.byRun[runId] && !force)) return;
  charts.loading[runId] = true;
  charts.error[runId] = '';
  try {
    charts.byRun[runId] = await getJSON('/runs/' + runId + '/charts', 'Could not load the charts');
  } catch (e) {
    charts.error[runId] = e.message;
  } finally {
    charts.loading[runId] = false;
  }
}
