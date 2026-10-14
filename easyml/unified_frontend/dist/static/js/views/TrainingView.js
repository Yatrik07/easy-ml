import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from '../lib/vue.js';
import { Chart } from '../lib/vue.js';
import { cssVar, resolvedTheme } from '../lib/theme.js';
import { logKind, parseTimestamp, stripLeadingEmoji } from '../lib/format.js';
import { taskLabel } from '../store/session.js';
import { confirmAction, navigate } from '../store/ui.js';
import { startTraining, stopTraining, training } from '../store/training.js';

// Live accuracy/loss curves for image models. The chart is created from store data when the view
// mounts and destroyed when it unmounts, so leaving and coming back never leaves a detached canvas.
const EpochChart = {
  setup() {
    const canvas = ref(null);
    let chart = null;

    function colors() {
      return {
        text: cssVar('--text-muted'), grid: cssVar('--chart-grid'),
        c1: cssVar('--chart-1'), c2: cssVar('--chart-2'), c3: cssVar('--chart-3'), c4: cssVar('--chart-4'),
      };
    }

    function build() {
      if (!canvas.value || !Chart) return;
      if (chart) chart.destroy();
      const c = colors();
      const epochs = training.epochs;
      chart = new Chart(canvas.value.getContext('2d'), {
        type: 'line',
        data: {
          labels: epochs.map(e => 'Epoch ' + e.epoch),
          datasets: [
            { label: 'Train accuracy', data: epochs.map(e => e.train_acc), borderColor: c.c1, backgroundColor: c.c1, yAxisID: 'y', tension: 0.3 },
            { label: 'Validation accuracy', data: epochs.map(e => e.val_acc), borderColor: c.c2, backgroundColor: c.c2, yAxisID: 'y', tension: 0.3 },
            { label: 'Train loss', data: epochs.map(e => e.train_loss), borderColor: c.c3, backgroundColor: c.c3, yAxisID: 'y1', borderDash: [5, 5], tension: 0.3 },
            { label: 'Validation loss', data: epochs.map(e => e.val_loss), borderColor: c.c4, backgroundColor: c.c4, yAxisID: 'y1', borderDash: [5, 5], tension: 0.3 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false,
          interaction: { mode: 'index', intersect: false },
          plugins: { legend: { labels: { color: c.text, boxWidth: 12 } } },
          scales: {
            x: { ticks: { color: c.text }, grid: { color: c.grid } },
            y: { position: 'left', min: 0, max: 1, title: { display: true, text: 'Accuracy', color: c.text }, ticks: { color: c.text }, grid: { color: c.grid } },
            y1: { position: 'right', title: { display: true, text: 'Loss', color: c.text }, ticks: { color: c.text }, grid: { drawOnChartArea: false } },
          },
        },
      });
    }

    onMounted(() => nextTick(build));
    watch(() => training.epochs.length, () => nextTick(build));
    watch(resolvedTheme, () => nextTick(build));
    onBeforeUnmount(() => { if (chart) chart.destroy(); chart = null; });
    return { canvas };
  },
  template: /* html */ `<div class="chart-box"><canvas ref="canvas" aria-label="Training curves"></canvas></div>`,
};

const LOG_ICONS = { success: 'check_circle', error: 'error', warning: 'warning_amber', info: 'radio_button_unchecked' };

export default {
  components: { EpochChart },
  setup() {
    const logBox = ref(null);
    const lines = computed(() => training.logs.map((entry, i) => {
      const kind = logKind(entry);
      const time = parseTimestamp(entry.timestamp);
      return {
        key: i,
        kind,
        icon: LOG_ICONS[kind],
        text: stripLeadingEmoji(entry.message) || entry.message,
        time: time ? time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
      };
    }));
    watch(() => training.logs.length, () => nextTick(() => {
      if (logBox.value) logBox.value.scrollTop = logBox.value.scrollHeight;
    }));

    const status = computed(() => ({
      running: { icon: 'model_training', title: training.stopping ? 'Stopping…' : 'Training in progress', text: 'You can browse other pages; training continues in the background.' },
      success: { icon: 'check_circle', title: 'Training complete', text: 'Your models are trained and saved.' },
      error: { icon: 'error', title: 'Training failed', text: (training.result && training.result.error) || 'See the log below for details.' },
      cancelled: { icon: 'stop_circle', title: 'Training stopped', text: 'The run was stopped before it finished.' },
      idle: { icon: 'hourglass_empty', title: 'No training run', text: 'Start a run from the Models step.' },
    }[training.status]));

    async function stop() {
      const ok = await confirmAction({
        title: 'Stop training?',
        message: 'The current run stops at the next checkpoint. Models that have not finished are discarded.',
        confirmLabel: 'Stop training',
        danger: true,
      });
      if (ok) stopTraining();
    }

    return { training, lines, logBox, status, stop, startTraining, navigate, taskLabel };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader :eyebrow="'Step 4 · ' + taskLabel(training.task)" title="Training" />

      <section class="card">
        <div class="card-body stack">
          <div class="row-between">
            <div class="training-status" :class="training.status">
              <div class="status-icon"><Icon :name="status.icon" /></div>
              <div>
                <h3>{{ status.title }}</h3>
                <div class="muted text-sm">{{ status.text }}</div>
              </div>
            </div>
            <div class="row">
              <button v-if="training.status === 'running'" class="btn btn-danger-outline" @click="stop" :disabled="training.stopping || !training.jobId">
                <Icon name="stop" />Stop
              </button>
              <template v-else-if="training.status === 'success'">
                <button class="btn btn-primary" @click="navigate('results')">View results<Icon name="arrow_forward" /></button>
              </template>
              <template v-else-if="training.status === 'error' || training.status === 'cancelled'">
                <button class="btn btn-secondary" @click="navigate('models')"><Icon name="arrow_back" />Back to models</button>
                <button class="btn btn-primary" @click="startTraining"><Icon name="replay" />Try again</button>
              </template>
            </div>
          </div>
          <div class="meter">
            <div class="progress"><div class="progress-bar" :class="{ success: training.status === 'success', danger: training.status === 'error' }"
                 :style="{ width: training.progress + '%' }"></div></div>
            <span class="num strong text-sm">{{ Math.round(training.progress) }}%</span>
          </div>
        </div>
      </section>

      <section v-if="training.epochs.length" class="card">
        <div class="card-header"><div><h3>Training curves</h3><div class="card-hint">Accuracy (solid) and loss (dashed) per epoch.</div></div></div>
        <div class="card-body"><EpochChart /></div>
      </section>

      <section class="card">
        <div class="card-header"><h3>Log</h3></div>
        <div ref="logBox" class="log" aria-live="polite">
          <div v-for="(line, i) in lines" :key="line.key" class="log-line" :class="[line.kind, { latest: i === lines.length - 1 && training.status === 'running' }]">
            <Icon :name="line.icon" />
            <span>{{ line.text }}</span>
            <span class="log-time">{{ line.time }}</span>
          </div>
          <div v-if="!lines.length" class="log-line"><Spinner /><span class="muted">Starting…</span></div>
        </div>
      </section>
    </div>`,
};
