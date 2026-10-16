// Results of a run: pipeline summary, ranked leaderboard and per-model detail.
// Used by the Results page and by the Run History detail drawer.
import { computed, ref } from '../lib/vue.js';
import {
  classLabels, formatMape, formatNumber, formatPercent, formatScore, metricFor, overfitHint, rankPerformance,
} from '../lib/format.js';
import { modelLabel } from '../store/session.js';

export const PipelineSummary = {
  props: { summary: Object, task: String },
  setup(props) {
    const steps = computed(() => {
      const s = props.summary || {};
      const out = [];
      const ds = s.dataset || {};
      if (props.task === 'ImageClassification') {
        const parts = [];
        if (ds.num_images != null) parts.push(formatNumber(ds.num_images) + ' images');
        if (ds.num_classes != null) parts.push(ds.num_classes + ' classes');
        out.push({ icon: 'photo_library', title: ds.name || 'Image folder', detail: parts.join(' · ') || 'Image dataset' });
      } else {
        out.push({ icon: 'table_view', title: ds.name || 'Dataset', detail: formatNumber(ds.rows) + ' rows · ' + ds.cols + ' columns' });
      }
      const pre = s.preprocessing || {};
      const preParts = [];
      if (pre.imputation) preParts.push('impute missing');
      if (pre.scaling) preParts.push('scale');
      if (pre.reduce_features) preParts.push('PCA');
      if (pre.augmentation) preParts.push('augmentation');
      if (pre.image_size) preParts.push(pre.image_size.join('×') + ' px');
      if (pre.lowercase) preParts.push('lowercase');
      if (pre.remove_stopwords) preParts.push('stopwords');
      if (pre.lemmatize) preParts.push('lemmatize');
      out.push({ icon: 'tune', title: 'Preprocessing', detail: preParts.join(' · ') || 'Defaults' });
      if (s.feature_engineering) {
        const fe = s.feature_engineering;
        out.push({ icon: 'data_object', title: 'Features', detail: fe.mode === 'embedding' ? 'Embeddings · ' + fe.embedding_model : (fe.vectorizer_type || 'tfidf').toUpperCase() });
      }
      if (s.imbalance_handling && s.imbalance_handling !== 'none') {
        out.push({ icon: 'balance', title: 'Rebalancing', detail: s.imbalance_handling.toUpperCase() });
      }
      out.push({ icon: 'call_split', title: 'Split', detail: s.split || '80/20' });
      out.push({ icon: 'model_training', title: 'Models', detail: (s.models_trained || []).length + ' trained' + (s.hyper_parameter_tuning ? ' · tuned' : '') });
      return out;
    });
    return { steps };
  },
  template: /* html */ `
    <div class="pipeline">
      <div v-for="step in steps" :key="step.title" class="pipeline-step">
        <Icon :name="step.icon" />
        <div class="truncate">
          <div class="step-title truncate">{{ step.title }}</div>
          <div class="step-detail truncate" :title="step.detail">{{ step.detail }}</div>
        </div>
      </div>
    </div>`,
};

export const ConfusionMatrix = {
  props: { entry: Object },
  setup(props) {
    const matrix = computed(() => (props.entry && props.entry.test_metrics && props.entry.test_metrics.confusion_matrix) || null);
    const labels = computed(() => {
      const names = classLabels(props.entry);
      return matrix.value && names.length === matrix.value.length ? names : (matrix.value || []).map((_, i) => 'Class ' + i);
    });
    const maxValue = computed(() => Math.max(1, ...(matrix.value || []).flat()));
    function cellStyle(value, isDiagonal) {
      const pct = Math.round((value / maxValue.value) * 85);
      const base = isDiagonal ? 'var(--success)' : 'var(--danger)';
      return {
        background: value ? 'color-mix(in srgb, ' + base + ' ' + Math.max(pct, 8) + '%, transparent)' : 'var(--surface-2)',
        color: pct > 55 ? 'var(--on-color)' : 'var(--text)',
      };
    }
    return { matrix, labels, cellStyle };
  },
  template: /* html */ `
    <div v-if="matrix" class="stack-sm">
      <div class="field-label">Confusion matrix <span class="subtle text-xs">(rows: actual, columns: predicted)</span></div>
      <div class="table-wrap flush">
        <table class="heatmap">
          <tr><th></th><th v-for="l in labels" :key="'c' + l" class="truncate">{{ l }}</th></tr>
          <tr v-for="(row, ri) in matrix" :key="ri">
            <th class="row-label truncate">{{ labels[ri] }}</th>
            <td v-for="(value, ci) in row" :key="ci" :style="cellStyle(value, ri === ci)">{{ value }}</td>
          </tr>
        </table>
      </div>
    </div>`,
};

export const ModelDetail = {
  props: { entry: Object, task: String, runId: [Number, String], modelKey: String },
  setup(props) {
    const report = computed(() => {
      const r = (props.entry && props.entry.test_metrics && props.entry.test_metrics.classification_report) || {};
      return Object.entries(r)
        .filter(([, v]) => v && typeof v === 'object' && v.precision !== undefined)
        .map(([name, v]) => ({ name, precision: v.precision, recall: v.recall, f1: v['f1-score'], support: v.support }));
    });
    const params = computed(() => Object.entries((props.entry && props.entry.best_params) || {}));
    const fmt = v => (v === null || v === undefined ? '—' : Number(v).toFixed(3));
    return { report, params, fmt };
  },
  template: /* html */ `
    <div class="grid-2">
      <div class="stack">
        <ConfusionMatrix :entry="entry" />
        <div v-if="params.length" class="stack-sm">
          <div class="field-label">Best hyperparameters</div>
          <dl class="kv"><template v-for="[k, v] in params" :key="k"><dt>{{ k }}</dt><dd>{{ v }}</dd></template></dl>
        </div>
        <p v-if="!params.length && task !== 'ImageClassification'" class="subtle text-sm">Trained with default hyperparameters.</p>
      </div>
      <div v-if="report.length" class="stack-sm">
        <div class="field-label">Per-class report (test set)</div>
        <div class="table-wrap">
          <table class="table table-compact">
            <thead><tr><th>Class</th><th class="right">Precision</th><th class="right">Recall</th><th class="right">F1</th><th class="right">Support</th></tr></thead>
            <tbody>
              <tr v-for="row in report" :key="row.name">
                <td>{{ row.name }}</td>
                <td class="right num">{{ fmt(row.precision) }}</td>
                <td class="right num">{{ fmt(row.recall) }}</td>
                <td class="right num">{{ fmt(row.f1) }}</td>
                <td class="right num">{{ row.support }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    <ModelCharts v-if="runId && modelKey" :run-id="runId" :model-key="modelKey" class="detail-charts" />`,
};

export const ResultsBoard = {
  props: {
    task: String,
    performance: { type: Object, default: () => ({}) },
    summary: Object,
    runId: [Number, String],
  },
  emits: ['predict'],
  setup(props) {
    const metric = computed(() => metricFor(props.task));
    const rows = computed(() => rankPerformance(props.task, props.performance, modelLabel));
    const isRegression = computed(() => props.task === 'MLRegression');
    const showF1 = computed(() => !isRegression.value && rows.value.some(r => r.f1 !== null));
    // k-fold cross-validation on the training data (runs trained with cross-validation on).
    const showCv = computed(() => rows.value.some(r => r.entry && r.entry.cv));
    function cvText(row) {
      const cv = row.entry && row.entry.cv;
      if (!cv || cv.mean === undefined || cv.mean === null) return '—';
      const spread = metric.value.kind === 'percent' ? formatNumber(cv.std * 100, 1) : formatNumber(cv.std, 3);
      return formatScore(cv.mean, metric.value) + ' ± ' + spread;
    }
    function cvTitle(row) {
      const cv = row.entry && row.entry.cv;
      if (!cv) return '';
      if (cv.error) return cv.error;
      return cv.folds + '-fold cross-validation on the training data: ' + cv.scores.map(s => formatScore(s, metric.value)).join(', ');
    }
    const expanded = ref(null);
    function toggle(key) { expanded.value = expanded.value === key ? null : key; }
    const hints = computed(() => rows.value
      .filter(r => overfitHint(r, metric.value))
      .map(r => ({ key: r.key, label: r.label, train: formatScore(r.train, metric.value), test: formatScore(r.test, metric.value) })));
    return {
      metric, rows, isRegression, showF1, showCv, cvText, cvTitle, expanded, toggle, hints, formatScore, formatNumber, formatMape, formatPercent,
    };
  },
  template: /* html */ `
    <div class="stack">
      <PipelineSummary v-if="summary" :summary="summary" :task="task" />

      <div class="card">
        <div class="card-header">
          <div>
            <h3>Leaderboard</h3>
            <div class="card-hint">Models ranked by test-set {{ metric.name }}. Select a model to see its details.</div>
          </div>
          <slot name="actions" />
        </div>
        <div class="table-wrap flush">
          <table class="table leaderboard">
            <thead>
              <tr>
                <th>#</th><th>Model</th>
                <th class="right">Test {{ metric.name }}</th>
                <th class="right">Train {{ metric.name }}</th>
                <th v-if="showCv" class="right" title="Mean ± spread over the cross-validation folds">CV {{ metric.name }}</th>
                <th v-if="showF1" class="right">Test F1</th>
                <th v-if="isRegression" class="right">MAE</th>
                <th v-if="isRegression" class="right">MAPE</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <template v-for="(row, index) in rows" :key="row.key">
                <tr class="clickable" :class="{ best: index === 0 && row.test !== null }" @click="toggle(row.key)"
                    tabindex="0" @keydown.enter="toggle(row.key)" :aria-expanded="expanded === row.key">
                  <td class="rank">{{ index + 1 }}</td>
                  <td><div class="model-cell">{{ row.label }}
                    <span v-if="index === 0 && row.test !== null" class="badge badge-accent"><Icon name="emoji_events" />Best</span></div></td>
                  <td class="right num strong">{{ formatScore(row.test, metric) }}</td>
                  <td class="right num muted">{{ formatScore(row.train, metric) }}</td>
                  <td v-if="showCv" class="right num nowrap" :title="cvTitle(row)">{{ cvText(row) }}</td>
                  <td v-if="showF1" class="right num">{{ row.f1 === null ? '—' : formatPercent(row.f1) }}</td>
                  <td v-if="isRegression" class="right num">{{ formatNumber(row.mae, 2) }}</td>
                  <td v-if="isRegression" class="right num">{{ formatMape(row.mape) }}</td>
                  <td class="right"><Icon :name="expanded === row.key ? 'expand_less' : 'expand_more'" /></td>
                </tr>
                <tr v-if="expanded === row.key" class="detail-row">
                  <td :colspan="5 + (showF1 ? 1 : 0) + (showCv ? 1 : 0) + (isRegression ? 2 : 0)">
                    <ModelDetail :entry="row.entry" :task="task" :run-id="runId" :model-key="row.key" />
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </div>

      <div v-if="hints.length" class="alert alert-warning">
        <Icon name="warning_amber" />
        <div class="stack-sm">
          <strong>Possible overfitting</strong>
          <span>These models score much higher on the training data than on the test data, so they may not generalize well:</span>
          <ul class="hint-list">
            <li v-for="hint in hints" :key="hint.key">{{ hint.label }}: train {{ hint.train }} vs test {{ hint.test }}</li>
          </ul>
        </div>
      </div>
    </div>`,
};

export const ClassDistribution = {
  props: { data: Object, loading: Boolean },
  setup(props) {
    const rows = computed(() => ((props.data && props.data.distribution) || []).slice(0, 12));
    const more = computed(() => Math.max(0, ((props.data && props.data.distribution) || []).length - 12));
    return { rows, more };
  },
  template: /* html */ `
    <div class="stack-sm">
      <div class="row-between">
        <div class="field-label">Target distribution</div>
        <span v-if="data && data.is_imbalanced" class="badge badge-warning"><Icon name="warning_amber" />Imbalanced</span>
        <span v-else-if="data" class="badge badge-success"><Icon name="check" />Balanced</span>
      </div>
      <div v-if="loading" class="muted text-sm">Checking class balance…</div>
      <div v-else-if="data" class="stack-sm">
        <div v-for="row in rows" :key="row.class" class="meter text-sm">
          <span class="meter-label truncate" :title="row.class">{{ row.class }}</span>
          <div class="progress"><div class="progress-bar" :style="{ width: row.percentage + '%' }"></div></div>
          <span class="meter-value num subtle">{{ row.percentage }}% · {{ row.count }}</span>
        </div>
        <div v-if="more" class="subtle text-xs">+{{ more }} more classes</div>
        <p v-if="data.is_imbalanced" class="field-hint">Consider SMOTE or Tomek Links below to rebalance the training data.</p>
      </div>
    </div>`,
};
