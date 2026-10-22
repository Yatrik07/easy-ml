// Results of a run: pipeline summary, ranked leaderboard and per-model detail.
// Used by the Results page and by the Run History detail drawer.
import { computed, ref, watch } from '../lib/vue.js';
import {
  classLabels, formatMape, formatNumber, formatPercent, formatScore, metricFor, overfitHint, rankPerformance,
} from '../lib/format.js';
import { modelLabel, session } from '../store/session.js';
import { charts, loadCharts } from '../store/charts.js';

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
      if (s.targets && s.targets.length > 1) {
        out.push({ icon: 'my_location', title: 'Predicting', detail: s.targets.join(', ') });
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
    // Several columns to predict: everything below is the chosen column's own model.
    const targets = computed(() => Object.keys((props.entry && props.entry.targets) || {}));
    const target = ref(null);
    watch(targets, list => { if (!list.includes(target.value)) target.value = list[0] || null; }, { immediate: true });
    const shown = computed(() => {
      const e = props.entry || {};
      if (!target.value) return e;
      const byTarget = (e.test_metrics && e.test_metrics.targets) || {};
      return { test_metrics: byTarget[target.value] || {}, best_params: (e.best_params || {})[target.value] || {} };
    });
    const metric = computed(() => metricFor(props.task));
    const targetFacts = computed(() => {
      if (!target.value) return [];
      const scores = props.entry.targets[target.value] || {};
      const facts = [
        { label: 'Test ' + metric.value.name, value: formatScore(scores.accuracy_test, metric.value) },
        { label: 'Train ' + metric.value.name, value: formatScore(scores.accuracy_train, metric.value) },
      ];
      if (scores.cv && scores.cv.mean != null) facts.push({ label: 'CV ' + metric.value.name, value: formatScore(scores.cv.mean, metric.value) });
      const tm = shown.value.test_metrics || {};
      if (tm.mae != null) facts.push({ label: 'MAE', value: formatNumber(tm.mae, 2) });
      if (tm.mape != null) facts.push({ label: 'MAPE', value: formatMape(tm.mape) });
      return facts;
    });
    const targetOverfit = computed(() => {
      if (!target.value) return null;
      const scores = props.entry.targets[target.value] || {};
      return overfitHint({ train: scores.accuracy_train, test: scores.accuracy_test }, metric.value);
    });
    const report = computed(() => {
      const r = (shown.value.test_metrics && shown.value.test_metrics.classification_report) || {};
      return Object.entries(r)
        .filter(([, v]) => v && typeof v === 'object' && v.precision !== undefined)
        .map(([name, v]) => ({ name, precision: v.precision, recall: v.recall, f1: v['f1-score'], support: v.support }));
    });
    const params = computed(() => Object.entries(shown.value.best_params || {}));
    const fmt = v => (v === null || v === undefined ? '—' : Number(v).toFixed(3));
    return { targets, target, shown, targetFacts, targetOverfit, report, params, fmt };
  },
  template: /* html */ `
    <div v-if="targets.length" class="segmented target-picker" role="group" aria-label="Column to predict">
      <button v-for="t in targets" :key="t" :class="{ active: target === t }" @click="target = t">{{ t }}</button>
    </div>
    <div class="grid-2">
      <div class="stack">
        <dl v-if="targetFacts.length" class="kv"><template v-for="f in targetFacts" :key="f.label"><dt>{{ f.label }}</dt><dd>{{ f.value }}</dd></template></dl>
        <p v-if="targetOverfit" class="field-hint">{{ targetOverfit }}</p>
        <ConfusionMatrix :entry="shown" />
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
    <ModelCharts v-if="runId && modelKey" :run-id="runId" :model-key="modelKey" :target="target" class="detail-charts" />`,
};

export const ResultsBoard = {
  props: {
    task: String,
    performance: { type: Object, default: () => ({}) },
    summary: Object,
    runId: [Number, String],
    targetWeights: Object, // several columns to predict: their weights (when there is no pipeline summary)
    featureWeights: Object, // input column weights (when there is no pipeline summary)
    targetNotes: Object, // several columns to predict: what the results mean per column (when no pipeline summary)
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
      if (!cv.scores) return cv.folds + '-fold cross-validation on the training data, weighted over the columns to predict';
      return cv.folds + '-fold cross-validation on the training data: ' + cv.scores.map(s => formatScore(s, metric.value)).join(', ');
    }
    const expanded = ref(null);
    function toggle(key) { expanded.value = expanded.value === key ? null : key; }
    // A final model (trained on every row) has no test set: its test scores are its evaluated run's.
    const estimatedFrom = computed(() => {
      const row = rows.value.find(r => r.entry && r.entry.estimated_from_run);
      return row ? row.entry.estimated_from_run : null;
    });
    const hints = computed(() => (estimatedFrom.value ? [] : rows.value
      .filter(r => overfitHint(r, metric.value))
      .map(r => ({ key: r.key, label: r.label, train: formatScore(r.train, metric.value), test: formatScore(r.test, metric.value) }))));

    // Several columns to predict: each has its own models; the leaderboard ranks their weighted average.
    const targets = computed(() => {
      const fromSummary = props.summary && props.summary.targets;
      if (fromSummary && fromSummary.length > 1) return fromSummary;
      const row = rows.value.find(r => r.entry && r.entry.targets);
      return row ? Object.keys(row.entry.targets) : [];
    });
    const weightOf = t => {
      const weights = props.targetWeights || (props.summary && props.summary.target_weights) || {};
      return weights[t] === undefined ? 1 : weights[t];
    };
    function targetScore(row, t) {
      const scores = row.entry && row.entry.targets && row.entry.targets[t];
      return scores && scores.accuracy_test != null ? scores.accuracy_test : null;
    }
    const winners = computed(() => Object.fromEntries(targets.value.map(t => {
      const scored = rows.value.filter(r => targetScore(r, t) !== null);
      if (!scored.length) return [t, null];
      return [t, scored.reduce((best, r) => (targetScore(r, t) > targetScore(best, t) ? r : best)).key];
    })));
    const showExact = computed(() => targets.value.length > 0 && rows.value.some(r => r.entry && r.entry.test_metrics
      && r.entry.test_metrics.exact_match != null));
    const exact = row => (row.entry && row.entry.test_metrics ? row.entry.test_metrics.exact_match : null);
    const showErrors = computed(() => isRegression.value && !targets.value.length);
    // Input column weights: how much each model then relied on the weighted columns (its importance chart).
    const featureWeights = computed(() => props.featureWeights || (props.summary && props.summary.feature_weights) || {});
    const weightedColumns = computed(() => Object.keys(featureWeights.value));
    watch(() => [props.runId, weightedColumns.value.length], ([id, count]) => { if (id && count) loadCharts(id); },
      { immediate: true });
    const EFFECTS = { follows: 'Yes', slight: 'Barely', none: 'No' };
    const weightRows = computed(() => rows.value.map(row => {
      const byClass = (session.capabilities && session.capabilities.weight_effects_by_class) || {};
      const saved = charts.byRun[props.runId];
      let model = saved && saved.models ? saved.models[row.key] : null;
      if (model && model.targets) model = model.targets[Object.keys(model.targets)[0]];
      const features = (model && model.importance && model.importance.features) || [];
      const total = features.reduce((sum, f) => sum + Math.max(0, f.value || 0), 0);
      const cells = weightedColumns.value.map(column => {
        const index = features.findIndex(f => f.name === column);
        if (index < 0 || !total) return '—';
        return formatPercent(Math.max(0, features[index].value) / total) + ' (#' + (index + 1) + ')';
      });
      return { key: row.key, label: row.label, effect: EFFECTS[byClass[row.key] || 'none'], cells };
    }));
    // Per column: not predictable from the inputs, overfitting, tied to another target (server-written text).
    const notes = computed(() => {
      const all = props.targetNotes || (props.summary && props.summary.target_notes) || {};
      return targets.value.flatMap(t => (all[t] || []).map(n => ({ key: t + n.kind, kind: n.kind, text: n.text })));
    });
    const rowsLeftOut = computed(() => (props.summary && props.summary.rows_without_targets) || 0);
    const colspan = computed(() => 5 + (showF1.value ? 1 : 0) + (showCv.value ? 1 : 0) + (showErrors.value ? 2 : 0)
      + (showExact.value ? 1 : 0));
    return {
      metric, rows, isRegression, showF1, showCv, cvText, cvTitle, expanded, toggle, hints, estimatedFrom,
      formatScore, formatNumber, formatMape, formatPercent, targets, weightOf, targetScore, winners, showExact, exact,
      showErrors, colspan, featureWeights, weightedColumns, weightRows, notes, rowsLeftOut,
    };
  },
  template: /* html */ `
    <div class="stack">
      <PipelineSummary v-if="summary" :summary="summary" :task="task" />

      <div v-if="estimatedFrom" class="alert alert-info">
        <Icon name="verified" />
        <div><strong>Final model, trained on all the data of run #{{ estimatedFrom }}.</strong>
          No rows were left to test it on, so its test and cross-validation scores (and charts) are run
          #{{ estimatedFrom }}'s, measured on data the model had not seen: expect about that score in use.
          Its train score is measured on all the rows.</div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h3>Leaderboard</h3>
            <div v-if="estimatedFrom" class="card-hint">Expected {{ metric.name }} from run #{{ estimatedFrom }}'s test set. Select a model to see its details.</div>
            <div v-else-if="targets.length" class="card-hint">Models ranked by the weighted average of their test-set {{ metric.name }} over {{ targets.length }} columns. Select a model to see each column's details.</div>
            <div v-else class="card-hint">Models ranked by test-set {{ metric.name }}. Select a model to see its details.</div>
          </div>
          <slot name="actions" />
        </div>
        <div class="table-wrap flush">
          <table class="table leaderboard">
            <thead>
              <tr>
                <th>#</th><th>Model</th>
                <th class="right" :title="estimatedFrom ? 'Test score of run #' + estimatedFrom : ''">{{ estimatedFrom ? 'Expected' : 'Test' }} {{ metric.name }}<template v-if="targets.length"> (weighted)</template></th>
                <th class="right">Train {{ metric.name }}</th>
                <th v-if="showCv" class="right" title="Mean ± spread over the cross-validation folds">CV {{ metric.name }}</th>
                <th v-if="showF1" class="right">Test F1</th>
                <th v-if="showExact" class="right" title="Test rows where every column was predicted right">All right</th>
                <th v-if="showErrors" class="right">MAE</th>
                <th v-if="showErrors" class="right">MAPE</th>
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
                  <td v-if="showExact" class="right num">{{ exact(row) === null ? '—' : formatPercent(exact(row)) }}</td>
                  <td v-if="showErrors" class="right num">{{ formatNumber(row.mae, 2) }}</td>
                  <td v-if="showErrors" class="right num">{{ formatMape(row.mape) }}</td>
                  <td class="right"><Icon :name="expanded === row.key ? 'expand_less' : 'expand_more'" /></td>
                </tr>
                <tr v-if="expanded === row.key" class="detail-row">
                  <td :colspan="colspan">
                    <ModelDetail :entry="row.entry" :task="task" :run-id="runId" :model-key="row.key" />
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </div>

      <div v-if="weightedColumns.length" class="card">
        <div class="card-header">
          <div>
            <h3>Column weights</h3>
            <div class="card-hint">How much each model relied on the columns you weighted: their share of the model's
              feature importance, and their rank among all columns<template v-if="targets.length"> (for {{ targets[0] }})</template>.
              Weights are hints; only some models can follow them.</div>
          </div>
        </div>
        <div class="table-wrap flush">
          <table class="table">
            <thead>
              <tr><th>Model</th><th>Follows weights</th>
                <th v-for="c in weightedColumns" :key="c" class="right">{{ c }} <span class="subtle text-xs">×{{ featureWeights[c] }}</span></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in weightRows" :key="row.key">
                <td>{{ row.label }}</td><td>{{ row.effect }}</td>
                <td v-for="(cell, i) in row.cells" :key="i" class="right num">{{ cell }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-if="targets.length" class="card">
        <div class="card-header">
          <div>
            <h3>By column to predict</h3>
            <div class="card-hint">Each column has its own models, trained and tuned on that column alone, so no column's
              accuracy is traded for another's. ★ marks each column's best model: together they make
              "Best for each target", which predictions use by default.</div>
          </div>
        </div>
        <div class="table-wrap flush">
          <table class="table target-scores">
            <thead>
              <tr><th>Model</th>
                <th v-for="t in targets" :key="t" class="right">{{ t }}<span v-if="weightOf(t) !== 1" class="subtle text-xs" :title="'Ranking weight ' + weightOf(t)"> ×{{ weightOf(t) }}</span></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in rows" :key="row.key">
                <td>{{ row.label }}</td>
                <td v-for="t in targets" :key="t" class="right num" :class="{ winner: winners[t] === row.key }">
                  {{ formatScore(targetScore(row, t), metric) }}<span v-if="winners[t] === row.key"> ★</span></td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="notes.length" class="card-body stack-sm">
          <div v-for="n in notes" :key="n.key" class="alert text-sm" :class="n.kind === 'depends_on_target' ? 'alert-info' : 'alert-warning'">
            <Icon :name="n.kind === 'depends_on_target' ? 'link' : 'warning_amber'" /><div>{{ n.text }}</div>
          </div>
          <p class="subtle text-xs">The ranking weights only rank the models; they do not change what any column's model learns.</p>
        </div>
        <div v-if="rowsLeftOut" class="card-body">
          <p class="subtle text-xs">{{ rowsLeftOut.toLocaleString() }} rows were left out because a column to predict was
            empty in them, so a single-column run on the same file trains and tests on more rows and its scores can differ.</p>
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
