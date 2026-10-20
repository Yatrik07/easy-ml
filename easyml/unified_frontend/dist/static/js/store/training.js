// Training runs: starts a background job, follows its progress over Socket.IO and keeps the result.
// The socket lives here (not in a view) so a run keeps streaming while the user browses other pages.
import { reactive } from '../lib/vue.js';
import { io } from '../lib/vue.js';
import { errorMessage } from '../lib/api.js';
import { navigate, toast, toastError, toastSuccess } from './ui.js';
import { wizard, isImage, isText, imageSize, severalTargets, targetColumns, targetWeight } from './wizard.js';
import { loadRuns } from './runs.js';

export const training = reactive({
  status: 'idle', // idle | running | success | error | cancelled
  jobId: null,
  logs: [],
  progress: 0,
  epochs: [],
  result: null,
  task: '',
  stopping: false,
  kind: 'wizard', // wizard | retrain | schedule | final
  parentRunId: null, // the run being retrained
  label: '', // the schedule's name (kind 'schedule')
  lastRequest: null, // { endpoint, payload, task, kind, parentRunId } for "Try again"
});

let socket = null;

function closeSocket() {
  if (socket) socket.disconnect();
  socket = null;
}

export function isRunning() {
  return training.status === 'running';
}

// The wizard's request for the current configuration.
function wizardRequest() {
  const config = wizard.config;
  let endpoint = '/ml_predict';
  const payload = { ...config, models: [...config.models] };
  delete payload.extra_targets;
  delete payload.target_weights;
  if (isText.value) {
    endpoint = '/nlp/predict';
  } else if (isImage.value) {
    endpoint = '/cv/predict';
    payload.image_size = imageSize();
    if (config.models.includes('CustomCNN')) payload.custom_layers = wizard.layers;
  } else {
    payload.preprocessing = true;
    if (severalTargets.value) {
      payload.target_columns = [...targetColumns.value];
      payload.target_weights = Object.fromEntries(targetColumns.value.map(c => [c, targetWeight(c)]));
      payload.imbalance_handling = 'none';
    }
  }
  return { endpoint, payload, task: config.task, kind: 'wizard', parentRunId: null };
}

export function startTraining() {
  return runJob(wizardRequest());
}

// Train a saved run's recipe on new data (POST /runs/<id>/retrain).
export function startRetrain(runId, body, task) {
  return runJob({ endpoint: '/runs/' + runId + '/retrain', payload: body, task, kind: 'retrain', parentRunId: runId });
}

// Train the final model of an evaluated run on all its rows (POST /runs/<id>/final).
export function startFinal(runId, body, task) {
  return runJob({ endpoint: '/runs/' + runId + '/final', payload: body, task, kind: 'final', parentRunId: runId });
}

// Run a retraining schedule now (POST /schedules/<id>/run); the schedule then keeps or replaces its model.
export function startScheduleRun(schedule) {
  return runJob({
    endpoint: '/schedules/' + schedule.id + '/run', payload: {}, task: schedule.task, kind: 'schedule',
    parentRunId: schedule.champion.id, label: schedule.name,
  });
}

export function retryTraining() {
  return training.lastRequest ? runJob(training.lastRequest) : startTraining();
}

// Start a background job and follow its progress until the final event.
async function runJob(request) {
  closeSocket();
  Object.assign(training, {
    status: 'running', jobId: null, logs: [], progress: 0, epochs: [], result: null,
    task: request.task, stopping: false, kind: request.kind, parentRunId: request.parentRunId, lastRequest: request,
    label: request.label || '',
  });
  navigate('training');

  const sock = io();
  socket = sock;
  // Events carry a per-job sequence number. Joining (or re-joining after a reconnect)
  // replays the job's history, so anything already shown is skipped.
  let jobId = null;
  let lastSeq = 0;
  const joinJob = () => {
    if (jobId && sock.connected) sock.emit('join_run', { job_id: jobId });
  };

  sock.on('progress', data => {
    if (data.job_id !== jobId) return;
    if (data.seq != null) {
      if (data.seq <= lastSeq) return;
      lastSeq = data.seq;
    }
    training.logs.push(data);
    if (data.pct !== null && data.pct !== undefined) training.progress = data.pct;
    else training.progress = Math.min(95, training.progress + 5);

    if (data.epoch_data && !data.epoch_data.early_stopped) training.epochs.push(data.epoch_data);

    if (data.done) finish(data.result || { run_id: -1 });
  });

  sock.on('connect_error', () => {
    if (training.status !== 'running') return;
    training.logs.push({ message: 'Lost connection to the Easy-ML server.', kind: 'error', done: true });
    finish({ status: 'error', error: 'Lost connection to the Easy-ML server. Is it still running?' });
  });

  // A (re)connect gets a new socket id, which is not in the job's room yet.
  sock.on('connect', joinJob);

  // Start the job only once the socket is connected, otherwise a fast run can finish
  // (and emit its final "done" event) before anyone is listening.
  await new Promise(resolve => {
    if (sock.connected) return resolve();
    sock.once('connect', resolve);
    setTimeout(resolve, 5000);
  });

  try {
    const res = await fetch(request.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request.payload),
    });
    if (!res.ok) throw new Error(await errorMessage(res, 'Failed to start training'));
    const started = await res.json();
    jobId = started.job_id;
    training.jobId = jobId;
    if (started.task) training.task = started.task;
    (started.warnings || []).forEach(message => training.logs.push({ message, kind: 'warning' }));
    // Joining replays everything the job emitted before this point.
    joinJob();
  } catch (e) {
    training.logs.push({ message: e.message, kind: 'error', done: true });
    finish({ status: 'error', error: e.message });
  }
}

function finish(result) {
  training.result = result;
  training.progress = 100;
  training.stopping = false;
  const status = result.status === 'error' ? 'error' : result.status === 'cancelled' ? 'cancelled' : 'success';
  training.status = status;
  if (status === 'error') toastError('Training failed: ' + (result.error || 'unknown error'));
  else if (status === 'cancelled') toast('Training was stopped.');
  else toastSuccess('Training complete. Your models are ready.');
  training.jobId = null;
  closeSocket();
  setTimeout(loadRuns, 800);
}

export async function stopTraining() {
  if (!training.jobId) {
    toastError('No training run to stop.');
    return;
  }
  try {
    const res = await fetch('/stop_training', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: training.jobId }),
    });
    if (!res.ok) throw new Error(await errorMessage(res, 'Failed to send stop signal'));
    training.stopping = true;
    training.logs.push({ message: 'Stop requested. Training halts at the next checkpoint…', kind: 'warning' });
  } catch (e) {
    toastError('Could not stop training: ' + e.message);
  }
}

export function resetTraining() {
  closeSocket();
  Object.assign(training, {
    status: 'idle', jobId: null, logs: [], progress: 0, epochs: [], result: null, task: '', stopping: false,
    kind: 'wizard', parentRunId: null, lastRequest: null,
  });
}
