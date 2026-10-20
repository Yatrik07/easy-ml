import { computed, ref } from '../lib/vue.js';
import { axis, seriesColor, useChart } from '../lib/charts.js';
import { absoluteTime, formatScore } from '../lib/format.js';
import { compare, swapCompare } from '../store/compare.js';
import { modelLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import { openInference } from '../store/inference.js';

function scoreDelta(a, b, metric) {
  if (a === null || a === undefined || b === null || b === undefined) return null;
  const diff = b - a;
  const text = metric.kind === 'percent'
    ? (diff >= 0 ? '+' : '−') + Math.abs(diff * 100).toFixed(1) + ' pts'
    : (diff >= 0 ? '+' : '−') + Math.abs(diff).toFixed(4);
  return { value: diff, text, better: diff > 0, worse: diff < 0 };
}

// Test scores of both runs per model, as grouped horizontal bars.
const ScoreCompareChart = {
  props: { rows: Array, metric: Object, labels: Array },
  setup(props) {
    const canvas = ref(null);
    useChart(canvas, colors => {
      if (!props.rows || !props.rows.length) return null;
      const scale = props.metric.kind === 'percent' ? 100 : 1;
      const value = v => (v === null || v === undefined ? null : v * scale);
      return {
        type: 'bar',
        data: {
          labels: props.rows.map(r => r.label),
          datasets: [
            { label: props.labels[0], data: props.rows.map(r => value(r.a)), backgroundColor: seriesColor(colors, 0) },
            { label: props.labels[1], data: props.rows.map(r => value(r.b)), backgroundColor: seriesColor(colors, 1) },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false, animation: false, indexAxis: 'y',
          plugins: { legend: { labels: { color: colors.text, boxWidth: 12 } } },
          scales: {
            x: axis(colors, 'Test ' + props.metric.name + (scale === 100 ? ' (%)' : ''), { beginAtZero: props.metric.kind === 'percent' }),
            y: axis(colors, '', { grid: { display: false } }),
          },
        },
      };
    }, () => props.rows);
    return { canvas };
  },
  template: /* html */ `<div class="chart-box"><canvas ref="canvas" role="img" aria-label="Test scores of both runs per model"></canvas></div>`,
};

export default {
  components: { ScoreCompareChart },
  setup() {
    const onlyDifferences = ref(false);
    const ready = computed(() => !!(compare.a && compare.b));
    const metric = computed(() => (ready.value ? compare.a.detail.metric : { name: 'Score', kind: 'percent' }));
    const names = computed(() => (ready.value ? ['#' + compare.a.detail.id + ' ' + compare.a.detail.run_name,
      '#' + compare.b.detail.id + ' ' + compare.b.detail.run_name] : ['', '']));

    const bestDelta = computed(() => (ready.value
      ? scoreDelta(compare.a.detail.best_score, compare.b.detail.best_score, metric.value) : null));

    const modelRows = computed(() => {
      if (!ready.value) return [];
      const scores = run => Object.fromEntries((run.detail.models || []).map(m => [m.key, m.score]));
      const a = scores(compare.a);
      const b = scores(compare.b);
      const keys = [...new Set(Object.keys(a).concat(Object.keys(b)))];
      return keys.map(key => ({
        key, label: modelLabel(key), a: key in a ? a[key] : null, b: key in b ? b[key] : null,
        onlyA: !(key in b), onlyB: !(key in a), delta: scoreDelta(a[key], b[key], metric.value),
      })).sort((x, y) => (y.b ?? y.a ?? -1) - (x.b ?? x.a ?? -1));
    });

    const settingRows = computed(() => {
      if (!ready.value) return [];
      const a = Object.fromEntries((compare.a.config.settings || []).map(s => [s.key, s]));
      const b = Object.fromEntries((compare.b.config.settings || []).map(s => [s.key, s]));
      const keys = [...new Set(Object.keys(a).concat(Object.keys(b)))];
      return keys.map(key => {
        const left = a[key] ? a[key].value : '—';
        const right = b[key] ? b[key].value : '—';
        return { key, label: (a[key] || b[key]).label, group: (a[key] || b[key]).group, a: left, b: right, differs: left !== right };
      });
    });
    const shownSettings = computed(() => settingRows.value.filter(r => !onlyDifferences.value || r.differs));
    const differenceCount = computed(() => settingRows.value.filter(r => r.differs).length
      + (ready.value && compare.a.detail.dataset_name !== compare.b.detail.dataset_name ? 1 : 0));

    return {
      compare, ready, metric, names, bestDelta, modelRows, settingRows, shownSettings, onlyDifferences, differenceCount,
      formatScore, absoluteTime, swapCompare, navigate, openInference,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader eyebrow="Run history" title="Compare runs"
                  :subtitle="ready ? 'Run #' + compare.b.detail.id + ' compared with run #' + compare.a.detail.id + '.' : ''">
        <button class="btn btn-secondary" @click="navigate('history')"><Icon name="arrow_back" />Run history</button>
        <button v-if="ready" class="btn btn-secondary" @click="swapCompare"><Icon name="swap_horiz" />Swap</button>
      </PageHeader>

      <div v-if="compare.loading" class="card card-body row muted"><Spinner /> Loading runs…</div>
      <div v-else-if="compare.error" class="alert alert-danger"><Icon name="error_outline" /><div>{{ compare.error }}</div></div>
      <div v-else-if="!ready" class="card"><EmptyState icon="compare_arrows" title="Nothing to compare"
           text="Select two runs of the same task in Run history and choose Compare." /></div>

      <template v-else>
        <div class="compare-grid">
          <div v-for="(side, i) in [compare.a, compare.b]" :key="side.detail.id" class="card card-body stack-sm">
            <div class="row-between">
              <span class="badge" :class="i === 0 ? '' : 'badge-accent'">{{ i === 0 ? 'A · baseline' : 'B' }}</span>
              <button class="btn btn-ghost btn-sm" @click="openInference(side.detail.id)"><Icon name="bolt" />Predict</button>
            </div>
            <h3>{{ side.detail.run_name }}</h3>
            <div class="run-sub"><span>#{{ side.detail.id }}</span><TaskBadge :task="side.detail.task" /><LineageBadge :run="side.detail" /><TargetsBadge :run="side.detail" /></div>
            <dl class="kv">
              <dt>Dataset</dt><dd>{{ side.detail.dataset_name || '—' }}</dd>
              <dt>Created</dt><dd>{{ absoluteTime(side.detail.timestamp) }}</dd>
              <dt>Best model</dt><dd>{{ side.detail.best_model_label || '—' }}</dd>
              <dt>Best {{ metric.name }}</dt>
              <dd class="row"><span class="strong">{{ formatScore(side.detail.best_score, metric) }}</span>
                <span v-if="i === 1 && bestDelta" class="badge" :class="{ 'badge-success': bestDelta.better, 'badge-danger': bestDelta.worse }">{{ bestDelta.text }}</span></dd>
            </dl>
          </div>
        </div>

        <section class="card">
          <div class="card-header"><div><h3>Model scores</h3><div class="card-hint">Test-set {{ metric.name }} of every model in either run.</div></div></div>
          <div class="card-body stack">
            <ScoreCompareChart :rows="modelRows" :metric="metric" :labels="names" />
            <div class="table-wrap flush">
              <table class="table table-compact">
                <thead><tr><th>Model</th><th class="right">A</th><th class="right">B</th><th class="right">Change</th></tr></thead>
                <tbody>
                  <tr v-for="row in modelRows" :key="row.key">
                    <td>{{ row.label }} <span v-if="row.onlyA" class="badge">only in A</span><span v-if="row.onlyB" class="badge">only in B</span></td>
                    <td class="right num">{{ formatScore(row.a, metric) }}</td>
                    <td class="right num">{{ formatScore(row.b, metric) }}</td>
                    <td class="right num"><span v-if="row.delta" :class="{ 'delta-up': row.delta.better, 'delta-down': row.delta.worse }">{{ row.delta.text }}</span><span v-else class="subtle">—</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section class="card">
          <div class="card-header">
            <div><h3>Settings</h3><div class="card-hint">{{ differenceCount }} difference{{ differenceCount === 1 ? '' : 's' }}, including the dataset.</div></div>
            <label class="row text-sm"><input type="checkbox" v-model="onlyDifferences" />Only differences</label>
          </div>
          <div class="table-wrap flush">
            <table class="table table-compact">
              <thead><tr><th>Setting</th><th>A</th><th>B</th></tr></thead>
              <tbody>
                <tr :class="{ 'diff-row': compare.a.detail.dataset_name !== compare.b.detail.dataset_name }"
                    v-if="!onlyDifferences || compare.a.detail.dataset_name !== compare.b.detail.dataset_name">
                  <td>Dataset</td><td>{{ compare.a.detail.dataset_name || '—' }}</td><td>{{ compare.b.detail.dataset_name || '—' }}</td>
                </tr>
                <tr v-for="row in shownSettings" :key="row.key" :class="{ 'diff-row': row.differs }">
                  <td>{{ row.label }} <span class="subtle text-xs">{{ row.group }}</span></td><td>{{ row.a }}</td><td>{{ row.b }}</td>
                </tr>
                <tr v-if="!settingRows.length"><td colspan="3" class="subtle">These runs were trained before settings were recorded (Easy-ML 0.6.0).</td></tr>
              </tbody>
            </table>
          </div>
        </section>
      </template>
    </div>`,
};
