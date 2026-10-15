// Result charts for one model: ROC, precision-recall, predicted vs actual, residuals, feature importance
// and top terms. Data comes from the run's evaluation.json (GET /runs/<id>/charts).
import { computed, onMounted, ref, watch } from '../lib/vue.js';
import { axis, seriesColor, useChart } from '../lib/charts.js';
import { formatNumber } from '../lib/format.js';
import { charts, loadCharts } from '../store/charts.js';

const BASE = { responsive: true, maintainAspectRatio: false, animation: false };

function points(xs, ys) {
  return (xs || []).map((x, i) => ({ x, y: ys[i] }));
}

function legend(colors) {
  return { labels: { color: colors.text, boxWidth: 12 } };
}

function curveChart(props, kind) {
  const canvas = ref(null);
  useChart(canvas, colors => {
    const data = props.data;
    if (!data || !data.curves) return null;
    const isRoc = kind === 'roc';
    const datasets = data.curves.map((curve, i) => ({
      label: curve.label + (isRoc ? ' (AUC ' + formatNumber(curve.auc, 3) + ')' : ' (AP ' + formatNumber(curve.ap, 3) + ')'),
      data: isRoc ? points(curve.fpr, curve.tpr) : points(curve.recall, curve.precision),
      borderColor: seriesColor(colors, i), backgroundColor: seriesColor(colors, i),
      showLine: true, pointRadius: 0, borderWidth: 2, stepped: isRoc ? false : 'after',
    }));
    if (isRoc && data.micro) {
      datasets.push({ label: 'Micro-average (AUC ' + formatNumber(data.micro_auc, 3) + ')', data: points(data.micro.fpr, data.micro.tpr),
        borderColor: colors.text, backgroundColor: colors.text, showLine: true, pointRadius: 0, borderWidth: 2, borderDash: [6, 4] });
    }
    if (isRoc) {
      datasets.push({ label: 'Random guess', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: colors.ref,
        backgroundColor: colors.ref, showLine: true, pointRadius: 0, borderWidth: 1, borderDash: [4, 4] });
    } else if (data.curves.length === 1) {
      const p = data.curves[0].prevalence;
      datasets.push({ label: 'Random guess', data: [{ x: 0, y: p }, { x: 1, y: p }], borderColor: colors.ref,
        backgroundColor: colors.ref, showLine: true, pointRadius: 0, borderWidth: 1, borderDash: [4, 4] });
    }
    return {
      type: 'scatter',
      data: { datasets },
      options: Object.assign({}, BASE, {
        plugins: { legend: legend(colors), tooltip: { enabled: false } },
        scales: {
          x: axis(colors, isRoc ? 'False positive rate' : 'Recall', { min: 0, max: 1 }),
          y: axis(colors, isRoc ? 'True positive rate' : 'Precision', { min: 0, max: 1 }),
        },
      }),
    };
  }, () => props.data);
  return { canvas };
}

export const RocChart = {
  props: { data: Object },
  setup(props) { return curveChart(props, 'roc'); },
  template: /* html */ `<div class="chart-box"><canvas ref="canvas" role="img" aria-label="ROC curve"></canvas></div>`,
};

export const PrChart = {
  props: { data: Object },
  setup(props) { return curveChart(props, 'pr'); },
  template: /* html */ `<div class="chart-box"><canvas ref="canvas" role="img" aria-label="Precision-recall curve"></canvas></div>`,
};

export const ScatterChart = {
  props: { data: Object },
  setup(props) {
    const canvas = ref(null);
    useChart(canvas, colors => {
      const data = props.data;
      if (!data || !data.actual) return null;
      const values = data.actual.concat(data.predicted).filter(v => v !== null);
      const lo = Math.min(...values);
      const hi = Math.max(...values);
      return {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Test rows', data: points(data.actual, data.predicted), backgroundColor: seriesColor(colors, 0), pointRadius: 3 },
            { label: 'Perfect prediction', data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: colors.ref,
              backgroundColor: colors.ref, showLine: true, pointRadius: 0, borderWidth: 1, borderDash: [4, 4] },
          ],
        },
        options: Object.assign({}, BASE, {
          plugins: { legend: legend(colors) },
          scales: { x: axis(colors, 'Actual'), y: axis(colors, 'Predicted') },
        }),
      };
    }, () => props.data);
    return { canvas };
  },
  template: /* html */ `<div class="chart-box"><canvas ref="canvas" role="img" aria-label="Predicted versus actual values"></canvas></div>`,
};

export const ResidualChart = {
  props: { data: Object },
  setup(props) {
    const canvas = ref(null);
    useChart(canvas, colors => {
      const data = props.data;
      if (!data || !data.counts) return null;
      const labels = data.counts.map((_, i) => formatNumber((data.edges[i] + data.edges[i + 1]) / 2, 2));
      return {
        type: 'bar',
        data: { labels, datasets: [{ label: 'Test rows', data: data.counts, backgroundColor: seriesColor(colors, 0) }] },
        options: Object.assign({}, BASE, {
          plugins: { legend: { display: false } },
          scales: { x: axis(colors, 'Error (actual − predicted)', { grid: { display: false } }), y: axis(colors, 'Rows') },
        }),
      };
    }, () => props.data);
    return { canvas };
  },
  template: /* html */ `<div class="chart-box"><canvas ref="canvas" role="img" aria-label="Distribution of prediction errors"></canvas></div>`,
};

export const ImportanceChart = {
  props: { data: Object },
  setup(props) {
    const canvas = ref(null);
    useChart(canvas, colors => {
      const data = props.data;
      if (!data || !data.features || !data.features.length) return null;
      const title = data.method === 'permutation'
        ? 'Score lost when the column is shuffled (' + (data.metric === 'r2' ? 'R²' : 'accuracy') + ')'
        : 'Share of the model’s importance';
      return {
        type: 'bar',
        data: {
          labels: data.features.map(f => f.name),
          datasets: [{ label: title, data: data.features.map(f => f.value), backgroundColor: seriesColor(colors, 0) }],
        },
        options: Object.assign({}, BASE, {
          indexAxis: 'y',
          plugins: { legend: { display: false } },
          scales: { x: axis(colors, title, { beginAtZero: true }), y: axis(colors, '', { grid: { display: false } }) },
        }),
      };
    }, () => props.data);
    const height = computed(() => (props.data && props.data.features && props.data.features.length > 10 ? 'chart-box tall' : 'chart-box'));
    return { canvas, height };
  },
  template: /* html */ `<div :class="height"><canvas ref="canvas" role="img" aria-label="Feature importance"></canvas></div>`,
};

// One prediction's SHAP breakdown: how much each input column moved the model's output, strongest first.
// Red (up) pushes towards the predicted class / a higher value, blue (down) away from it.
export function contributionText(value, scale) {
  const sign = value > 0 ? '+' : (value < 0 ? '−' : '');
  const size = Math.abs(Number(value) || 0);
  return scale === 'probability' ? sign + formatNumber(size * 100, 1) + ' pts' : sign + formatNumber(size, 3);
}

export const ContributionChart = {
  props: { data: Object, axisTitle: String },
  setup(props) {
    const canvas = ref(null);
    const items = computed(() => {
      const data = props.data;
      if (!data || !data.features) return [];
      const rows = data.features.map(f => ({
        label: data.grouped ? f.name + ' = ' + (f.missing ? 'missing' : String(f.value)) : f.name,
        value: f.contribution,
      }));
      if (data.other) {
        rows.push({ label: data.other.count + ' other column' + (data.other.count === 1 ? '' : 's'), value: data.other.contribution });
      }
      return rows;
    });
    useChart(canvas, colors => {
      if (!items.value.length) return null;
      const scale = props.data.scale;
      const up = seriesColor(colors, 2);
      const down = seriesColor(colors, 0);
      return {
        type: 'bar',
        data: {
          labels: items.value.map(i => i.label),
          datasets: [{
            data: items.value.map(i => i.value),
            backgroundColor: items.value.map(i => (i.value >= 0 ? up : down)),
            borderRadius: 3, maxBarThickness: 26,
          }],
        },
        options: Object.assign({}, BASE, {
          indexAxis: 'y',
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: ctx => contributionText(ctx.parsed.x, scale) } },
          },
          scales: {
            x: axis(colors, props.axisTitle, {
              ticks: { color: colors.text, callback: v => contributionText(v, scale) },
              grid: { color: ctx => (ctx.tick && ctx.tick.value === 0 ? colors.ref : colors.grid) },
            }),
            y: axis(colors, '', { grid: { display: false }, ticks: { color: colors.text, font: { size: 13 } } }),
          },
        }),
      };
    }, () => props.data);
    const boxClass = computed(() => (items.value.length > 8 ? 'chart-box tall' : 'chart-box'));
    return { canvas, boxClass };
  },
  template: /* html */ `<div :class="boxClass"><canvas ref="canvas" role="img" aria-label="How each input changed the prediction"></canvas></div>`,
};

export const TopTerms = {
  props: { data: Object },
  template: /* html */ `
    <div class="term-groups">
      <div v-for="group in (data.by_class || [])" :key="group.label" class="stack-xs">
        <div class="field-label">Words that point to “{{ group.label }}”</div>
        <div class="term-list"><span v-for="t in group.terms" :key="t.term" class="badge">{{ t.term }}</span></div>
      </div>
      <div v-if="data.overall" class="stack-xs">
        <div class="field-label">Most important words</div>
        <div class="term-list"><span v-for="t in data.overall" :key="t.term" class="badge">{{ t.term }}</span></div>
      </div>
    </div>`,
};

const TAB_LABELS = {
  roc: 'ROC curve', pr: 'Precision-recall', scatter: 'Predicted vs actual', residual_histogram: 'Errors',
  importance: 'Feature importance', top_terms: 'Top words',
};
const TAB_ORDER = ['roc', 'pr', 'scatter', 'residual_histogram', 'importance', 'top_terms'];

export const ModelCharts = {
  props: { runId: [Number, String], modelKey: String },
  setup(props) {
    onMounted(() => loadCharts(props.runId));
    watch(() => props.runId, id => loadCharts(id));
    const response = computed(() => charts.byRun[props.runId]);
    const model = computed(() => (response.value && response.value.models && response.value.models[props.modelKey]) || null);
    const tabs = computed(() => (model.value ? TAB_ORDER.filter(key => model.value[key]) : []));
    const active = ref(null);
    watch(tabs, list => { if (!list.includes(active.value)) active.value = list[0] || null; }, { immediate: true });
    const skipped = computed(() => Object.entries((model.value && model.value.skipped) || {})
      .map(([key, reason]) => ({ key, label: TAB_LABELS[key] || key, reason })));
    const summary = computed(() => {
      const m = model.value;
      if (!m) return '';
      const parts = [];
      if (m.roc) parts.push('ROC AUC ' + formatNumber(m.roc.micro_auc != null ? m.roc.micro_auc : m.roc.macro_auc, 3));
      if (m.pr && m.pr.curves.length === 1) parts.push('average precision ' + formatNumber(m.pr.curves[0].ap, 3));
      if (m.residual_histogram) parts.push('mean error ' + formatNumber(m.residual_histogram.mean, 3));
      return parts.join(' · ');
    });
    const decisionScores = computed(() => !!(model.value && model.value.roc && model.value.roc.score_source === 'decision'));
    return { charts, response, model, tabs, active, skipped, summary, decisionScores, TAB_LABELS, formatNumber };
  },
  template: /* html */ `
    <div class="stack-sm model-charts">
      <div class="row-between">
        <div class="field-label">Charts <span v-if="summary" class="subtle text-xs">({{ summary }})</span></div>
      </div>
      <div v-if="charts.loading[runId]" class="row muted text-sm"><Spinner /> Loading charts…</div>
      <div v-else-if="charts.error[runId]" class="muted text-sm">{{ charts.error[runId] }}</div>
      <p v-else-if="response && !response.available" class="chart-note">{{ response.reason }}</p>
      <p v-else-if="response && !model" class="chart-note">No charts were saved for this model.</p>
      <template v-else-if="model">
        <div v-if="tabs.length" class="tabs" role="tablist">
          <button v-for="key in tabs" :key="key" role="tab" :aria-selected="active === key"
                  :class="{ active: active === key }" @click="active = key">{{ TAB_LABELS[key] }}</button>
        </div>
        <RocChart v-if="active === 'roc'" :data="model.roc" />
        <PrChart v-else-if="active === 'pr'" :data="model.pr" />
        <ScatterChart v-else-if="active === 'scatter'" :data="model.scatter" />
        <ResidualChart v-else-if="active === 'residual_histogram'" :data="model.residual_histogram" />
        <template v-else-if="active === 'importance'">
          <ImportanceChart :data="model.importance" />
          <p v-if="model.importance.note" class="chart-note">{{ model.importance.note }}</p>
          <p v-if="model.importance.total_features > model.importance.features.length" class="chart-note">
            Showing the top {{ model.importance.features.length }} of {{ model.importance.total_features }} columns.</p>
        </template>
        <TopTerms v-else-if="active === 'top_terms'" :data="model.top_terms" />
        <p v-if="decisionScores && (active === 'roc' || active === 'pr')" class="chart-note">
          This model gives decision scores rather than probabilities; the curves use those scores.</p>
        <p v-if="model.scatter && model.scatter.sampled && active === 'scatter'" class="chart-note">
          Showing {{ model.scatter.actual.length }} of {{ formatNumber(model.scatter.n) }} test rows.</p>
        <ul v-if="skipped.length" class="chart-note hint-list">
          <li v-for="s in skipped" :key="s.key">{{ s.label }}: {{ s.reason }}</li>
        </ul>
      </template>
    </div>`,
};
