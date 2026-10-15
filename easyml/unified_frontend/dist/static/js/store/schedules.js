// Scheduled retraining: the list of schedules, the create/edit dialog, each schedule's history,
// and notifications when a schedule finishes (also while another page is open).
import { reactive } from '../lib/vue.js';
import { deleteJSON, getJSON, postJSON, putJSON } from '../lib/api.js';
import { confirmAction, toast, toastError, toastSuccess } from './ui.js';
import { loadRuns, runs } from './runs.js';
import { isRunning, startScheduleRun } from './training.js';

export const TRIGGERS = [
  { id: 'interval', label: 'Every few hours' },
  { id: 'daily', label: 'Daily' },
  { id: 'weekly', label: 'Weekly' },
  { id: 'new_data', label: 'When new data arrives' },
];
export const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export const schedules = reactive({
  list: [],
  scheduler: { studio: false, poll_seconds: 30 },
  loading: false,
  loaded: false,
  dialog: { open: false, editing: null, form: emptyForm(), check: null, checking: false, saving: false, recipes: [] },
  events: { open: false, schedule: null, list: [], loading: false },
});

function emptyForm() {
  return {
    run_id: '', name: '', source_path: '', file_pattern: '', trigger: 'daily', interval_hours: 6, at_time: '02:00',
    weekday: 0, recipe: '', margin: 0, enabled: true,
  };
}

export async function loadSchedules() {
  schedules.loading = true;
  try {
    const data = await getJSON('/schedules', 'Could not load the schedules');
    schedules.list = data.schedules;
    schedules.scheduler = data.scheduler;
    schedules.loaded = true;
  } catch (e) {
    toastError(e.message);
  } finally {
    schedules.loading = false;
  }
}

// --------------------------------------------------------------------------- create / edit dialog

export function runFor(runId) {
  return runs.list.find(r => String(r.id) === String(runId)) || null;
}

// The improvement margin is entered in the metric's own unit: accuracy points (1 = 1 %) or R².
export function marginInPoints(task) {
  return task !== 'MLRegression';
}

export async function openScheduleDialog(runId) {
  if (!runs.list.length) await loadRuns();
  const form = emptyForm();
  if (runId) {
    form.run_id = String(runId);
    const run = runFor(runId);
    form.name = run ? (run.run_name + '-schedule').slice(0, 80) : '';
  }
  Object.assign(schedules.dialog, { open: true, editing: null, form, check: null, checking: false, saving: false });
  loadDialogRecipes();
}

export function openEditDialog(schedule) {
  const form = {
    run_id: String(schedule.base_run.id), name: schedule.name, source_path: schedule.source_path,
    file_pattern: schedule.file_pattern || '', trigger: schedule.trigger,
    interval_hours: schedule.interval_minutes ? +(schedule.interval_minutes / 60).toFixed(2) : 6,
    at_time: schedule.at_time || '02:00', weekday: schedule.weekday || 0, recipe: schedule.recipe || '',
    margin: marginInPoints(schedule.task) ? +(schedule.min_improvement * 100).toFixed(2) : schedule.min_improvement,
    enabled: schedule.enabled,
  };
  Object.assign(schedules.dialog, { open: true, editing: schedule, form, check: null, checking: false, saving: false });
  loadDialogRecipes();
  checkSource();
}

export function closeScheduleDialog() {
  schedules.dialog.open = false;
}

async function loadDialogRecipes() {
  try {
    schedules.dialog.recipes = (await getJSON('/prep/recipes', 'Could not load recipes')).recipes || [];
  } catch (e) {
    schedules.dialog.recipes = [];
  }
}

let checkTimer = null;
let checkSeq = 0;

// Look at the source as the schedule would (debounced while typing).
export function checkSource(delay = 0) {
  clearTimeout(checkTimer);
  const form = schedules.dialog.form;
  if (!form.run_id || !form.source_path.trim()) {
    schedules.dialog.check = null;
    return;
  }
  checkTimer = setTimeout(async () => {
    const seq = ++checkSeq;
    schedules.dialog.checking = true;
    try {
      const result = await postJSON('/schedules/check-source', {
        run_id: Number(form.run_id), source_path: form.source_path, file_pattern: form.file_pattern,
      }, 'Could not check the data source');
      if (seq === checkSeq) schedules.dialog.check = result;
    } catch (e) {
      if (seq === checkSeq) schedules.dialog.check = { ok: false, message: e.message };
    } finally {
      if (seq === checkSeq) schedules.dialog.checking = false;
    }
  }, delay);
}

function payload() {
  const form = schedules.dialog.form;
  const run = runFor(form.run_id);
  const task = run ? run.task : (schedules.dialog.editing && schedules.dialog.editing.task);
  const margin = Number(form.margin) || 0;
  const body = {
    name: form.name.trim(), source_path: form.source_path.trim(), file_pattern: form.file_pattern.trim(),
    trigger: form.trigger, at_time: form.at_time, weekday: Number(form.weekday),
    interval_minutes: Math.round(Number(form.interval_hours) * 60), recipe: form.recipe,
    min_improvement: marginInPoints(task) ? margin / 100 : margin, enabled: form.enabled,
  };
  if (!schedules.dialog.editing) body.run_id = Number(form.run_id);
  return body;
}

export async function saveSchedule() {
  const dialog = schedules.dialog;
  dialog.saving = true;
  try {
    const saved = dialog.editing
      ? await putJSON('/schedules/' + dialog.editing.id, payload(), 'Could not save the schedule')
      : await postJSON('/schedules', payload(), 'Could not create the schedule');
    dialog.open = false;
    toastSuccess((dialog.editing ? 'Saved' : 'Created') + ' schedule "' + saved.name + '".');
    loadSchedules();
  } catch (e) {
    toastError(e.message);
  } finally {
    dialog.saving = false;
  }
}

// --------------------------------------------------------------------------- actions

export async function toggleSchedule(schedule) {
  try {
    const saved = await postJSON('/schedules/' + schedule.id + '/toggle', { enabled: !schedule.enabled },
      'Could not change the schedule');
    Object.assign(schedule, saved);
  } catch (e) {
    toastError(e.message);
  }
}

export async function runScheduleNow(schedule) {
  if (isRunning()) {
    toastError('Another run is still training. Wait for it to finish or stop it first.');
    return;
  }
  await startScheduleRun(schedule);
}

export async function deleteSchedule(schedule) {
  const ok = await confirmAction({
    title: 'Delete schedule "' + schedule.name + '"?',
    message: 'It stops retraining. The runs it trained stay in Run history.',
    confirmLabel: 'Delete schedule',
    danger: true,
  });
  if (!ok) return;
  try {
    await deleteJSON('/schedules/' + schedule.id, {}, 'Could not delete the schedule');
    loadSchedules();
  } catch (e) {
    toastError(e.message);
  }
}

export async function openEvents(schedule) {
  Object.assign(schedules.events, { open: true, schedule, list: [], loading: true });
  try {
    const data = await getJSON('/schedules/' + schedule.id, 'Could not load the history');
    schedules.events.list = data.events;
  } catch (e) {
    toastError(e.message);
  } finally {
    schedules.events.loading = false;
  }
}

export function closeEvents() {
  schedules.events.open = false;
}

// --------------------------------------------------------------------------- notifications

export function eventText(event) {
  const name = 'Schedule "' + (event.schedule_name || '') + '"';
  if (event.decision === 'promoted') return name + ': new run #' + event.challenger_run_id + ' is now in use.';
  if (event.decision === 'kept champion') return name + ': run #' + event.champion_after + ' stays in use; the new run was not better.';
  return name + ' failed: ' + (event.message || 'unknown error');
}

let listening = false;

// One quiet connection for the whole app: a toast and fresh lists when any schedule finishes.
export function listenForScheduleEvents() {
  if (listening || typeof io !== 'function') return;
  listening = true;
  const sock = io();
  sock.on('schedule_event', event => {
    if (event.decision === 'failed') toastError(eventText(event));
    else toast(eventText(event), event.decision === 'promoted' ? 'success' : 'info');
    if (schedules.loaded) loadSchedules();
    if (schedules.events.open && schedules.events.schedule && schedules.events.schedule.id === event.schedule_id) {
      openEvents(schedules.events.schedule);
    }
    loadRuns();
  });
}
