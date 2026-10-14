// Training runs: starts a background job, follows its progress over Socket.IO and keeps the result.
// The socket lives here (not in a view) so a run keeps streaming while the user browses other pages.
import { reactive } from '../lib/vue.js';
import { io } from '../lib/vue.js';
import { errorMessage } from '../lib/api.js';
import { navigate, toast, toastError, toastSuccess } from './ui.js';
import { wizard, isImage, isText, imageSize } from './wizard.js';
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
});

let socket = null;

function closeSocket() {
  if (socket) socket.disconnect();
  socket = null;
}

export function isRunning() {
  return training.status === 'running';
}

export async function startTraining() {
  closeSocket();
  Object.assign(training, {
    status: 'running', jobId: null, logs: [], progress: 0, epochs: [], result: null,
    task: wizard.config.task, stopping: false,
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
    const config = wizard.config;
    let endpoint = '/ml_predict';
    const payload = { ...config, models: [...config.models] };
    if (isText.value) {
      endpoint = '/nlp/predict';
    } else if (isImage.value) {
      endpoint = '/cv/predict';
      payload.image_size = imageSize();
      if (config.models.includes('CustomCNN')) payload.custom_layers = wizard.layers;
    } else {
      payload.preprocessing = true;
    }
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await errorMessage(res, 'Failed to start training'));
    jobId = (await res.json()).job_id;
    training.jobId = jobId;
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
  });
}
