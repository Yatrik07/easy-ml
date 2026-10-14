// Small presentational components used across views.
// Templates are plain strings compiled by Vue at runtime: no `${` and no backticks inside them.
import { computed } from '../lib/vue.js';
import { formatScore, TASK_ICONS } from '../lib/format.js';
import { taskLabel } from '../store/session.js';

export const Icon = {
  props: { name: { type: String, required: true }, size: { type: String, default: '' } },
  template: /* html */ `<span class="icon" :class="size ? 'icon-' + size : ''" aria-hidden="true">{{ name }}</span>`,
};

export const PageHeader = {
  props: { title: String, subtitle: String, eyebrow: String },
  template: /* html */ `
    <header class="page-header">
      <div>
        <div v-if="eyebrow" class="eyebrow">{{ eyebrow }}</div>
        <h1>{{ title }}</h1>
        <p v-if="subtitle" class="page-subtitle">{{ subtitle }}</p>
      </div>
      <div class="row"><slot /></div>
    </header>`,
};

export const TaskBadge = {
  props: { task: String },
  setup(props) {
    return { label: computed(() => taskLabel(props.task)), icon: computed(() => TASK_ICONS[props.task] || 'science') };
  },
  template: /* html */ `<span class="badge"><Icon :name="icon" />{{ label }}</span>`,
};

export const ModelChips = {
  props: { models: { type: Array, default: () => [] }, max: { type: Number, default: 2 } },
  setup(props) {
    const shown = computed(() => props.models.slice(0, props.max));
    const hidden = computed(() => props.models.slice(props.max));
    const hiddenTitle = computed(() => hidden.value.map(m => m.label).join(', '));
    return { shown, hidden, hiddenTitle };
  },
  template: /* html */ `
    <div class="chip-list">
      <span v-for="m in shown" :key="m.key" class="chip">{{ m.label }}</span>
      <span v-if="hidden.length" class="chip" :title="hiddenTitle">+{{ hidden.length }}</span>
    </div>`,
};

export const ScoreValue = {
  props: { value: Number, metric: Object },
  setup(props) {
    return { text: computed(() => formatScore(props.value, props.metric)) };
  },
  template: /* html */ `<span class="num">{{ text }}</span>`,
};

export const EmptyState = {
  props: { icon: { type: String, default: 'inbox' }, title: String, text: String },
  template: /* html */ `
    <div class="empty">
      <Icon :name="icon" />
      <h3>{{ title }}</h3>
      <p v-if="text">{{ text }}</p>
      <div class="row"><slot /></div>
    </div>`,
};

export const StatTile = {
  props: { label: String, value: [String, Number] },
  template: /* html */ `
    <div class="stat">
      <div class="stat-label">{{ label }}</div>
      <div class="stat-value">{{ value }}</div>
    </div>`,
};

export const Spinner = { template: /* html */ `<span class="spinner" role="status" aria-label="Loading"></span>` };
