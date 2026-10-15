// Data prep building blocks: step forms (generated from the server's step specs), the step list,
// the preview table, the column profile, and the add-step / recipe / train dialogs.
import { computed, ref, watch } from '../lib/vue.js';
import { formatNumber, truncate } from '../lib/format.js';
import { featureAvailable } from '../store/session.js';
import {
  addStep, cleanFileName, columnsBefore, deleteRecipe, loadRecipes, moveStep, prep, removeStep, saveRecipe,
  specFor, toggleStep, trainWithData, updateParam, useRecipe,
} from '../store/prep.js';

const KIND_ICONS = { number: 'tag', text: 'notes', category: 'label', date: 'event', boolean: 'toggle_on' };

function visible(field, params) {
  return !field.show_if || params[field.show_if[0]] === field.show_if[1];
}

// Old value → new value rows, kept locally so a half-typed row does not disappear.
export const MappingField = {
  props: { index: Number, field: Object, columns: Array },
  setup(props) {
    const initial = Object.entries(prep.steps[props.index].params[props.field.name] || {});
    const rows = ref(initial.length ? initial.map(([k, v]) => ({ k, v })) : [{ k: '', v: '' }]);
    function sync() {
      const mapping = {};
      for (const row of rows.value) if (String(row.k).trim() !== '') mapping[row.k] = row.v;
      updateParam(props.index, props.field.name, mapping);
    }
    function add() { rows.value.push({ k: '', v: '' }); }
    function remove(i) { rows.value.splice(i, 1); sync(); }
    return { rows, sync, add, remove };
  },
  template: /* html */ `
    <div class="stack-xs">
      <div v-for="(row, i) in rows" :key="i" class="mapping-row">
        <select v-if="field.key === 'column'" class="select" v-model="row.k" @change="sync" aria-label="Column">
          <option value="">Column…</option>
          <option v-for="c in columns" :key="c" :value="c">{{ c }}</option>
        </select>
        <input v-else class="input" v-model="row.k" @input="sync" placeholder="Value" aria-label="Value to replace" />
        <Icon name="arrow_forward" />
        <input class="input" v-model="row.v" @input="sync" :placeholder="field.key === 'column' ? 'New name' : 'Replacement (empty = missing)'"
               aria-label="Replacement" />
        <button class="btn btn-ghost btn-icon" @click="remove(i)" aria-label="Remove row"><Icon name="close" /></button>
      </div>
      <button class="btn btn-ghost btn-sm" @click="add"><Icon name="add" />Add a row</button>
    </div>`,
};

export const StepForm = {
  props: { index: Number },
  setup(props) {
    const step = computed(() => prep.steps[props.index]);
    const spec = computed(() => specFor(step.value.op));
    const columns = computed(() => columnsBefore(props.index));
    const fields = computed(() => spec.value.fields.filter(f => visible(f, step.value.params)));
    function set(name, value) { updateParam(props.index, name, value); }
    function toggleIn(name, value) {
      const list = [...(step.value.params[name] || [])];
      const at = list.indexOf(value);
      if (at >= 0) list.splice(at, 1);
      else list.push(value);
      set(name, list);
    }
    function has(name, value) { return (step.value.params[name] || []).includes(value); }
    return { step, spec, columns, fields, set, toggleIn, has };
  },
  template: /* html */ `
    <div class="stack-sm">
      <p class="subtle text-xs">{{ spec.help }}</p>
      <div v-for="field in fields" :key="field.name" class="field">
        <label class="field-label">{{ field.label }}</label>
        <div v-if="field.type === 'columns'" class="chip-list">
          <button v-for="c in columns" :key="c" class="chip" :class="{ selected: has(field.name, c) }"
                  :aria-pressed="has(field.name, c)" @click="toggleIn(field.name, c)">{{ c }}</button>
        </div>
        <div v-else-if="field.type === 'parts'" class="chip-list">
          <button v-for="o in field.options" :key="o[0]" class="chip" :class="{ selected: has(field.name, o[0]) }"
                  :aria-pressed="has(field.name, o[0])" @click="toggleIn(field.name, o[0])">{{ o[1] }}</button>
        </div>
        <select v-else-if="field.type === 'column'" class="select" :value="step.params[field.name] || ''"
                @change="set(field.name, $event.target.value)">
          <option value="">Choose a column…</option>
          <option v-for="c in columns" :key="c" :value="c">{{ c }}</option>
        </select>
        <select v-else-if="field.type === 'choice'" class="select" :value="step.params[field.name]"
                @change="set(field.name, $event.target.value)">
          <option v-for="o in field.options" :key="o[0]" :value="o[0]">{{ o[1] }}</option>
        </select>
        <label v-else-if="field.type === 'bool'" class="row text-sm">
          <input type="checkbox" :checked="!!step.params[field.name]" @change="set(field.name, $event.target.checked)" />
          <span>Yes</span>
        </label>
        <input v-else-if="field.type === 'number'" class="input" type="number" step="any" :value="step.params[field.name]"
               @input="set(field.name, $event.target.value === '' ? '' : Number($event.target.value))" />
        <MappingField v-else-if="field.type === 'mapping'" :key="index + field.name" :index="index" :field="field" :columns="columns" />
        <input v-else class="input" :value="step.params[field.name]" @input="set(field.name, $event.target.value)" />
      </div>
    </div>`,
};

function describe(step, spec) {
  const parts = [];
  for (const field of spec.fields) {
    if (!visible(field, step.params)) continue;
    const value = step.params[field.name];
    if (value === '' || value === null || value === undefined) continue;
    if (Array.isArray(value)) {
      if (value.length) parts.push(value.slice(0, 3).join(', ') + (value.length > 3 ? ' +' + (value.length - 3) : ''));
    } else if (field.type === 'mapping') {
      const n = Object.keys(value).length;
      if (n) parts.push(n + ' change' + (n === 1 ? '' : 's'));
    } else if (field.type === 'choice') {
      const option = field.options.find(o => o[0] === value);
      parts.push(option ? option[1] : String(value));
    } else if (field.type === 'bool') {
      if (value) parts.push(field.label.toLowerCase());
    } else {
      parts.push(String(value));
    }
  }
  return parts.join(' · ');
}

export const StepList = {
  setup() {
    const items = computed(() => prep.steps.map((step, index) => {
      const spec = specFor(step.op);
      const report = (prep.result && prep.result.steps && prep.result.steps[index]) || null;
      return { index, step, spec, summary: spec ? describe(step, spec) : '', report };
    }));
    function open(index) { prep.openStep = prep.openStep === index ? -1 : index; }
    return { prep, items, open, toggleStep, moveStep, removeStep, formatNumber };
  },
  template: /* html */ `
    <ol class="step-list">
      <li v-for="item in items" :key="item.index" class="step-card"
          :class="{ disabled: !item.step.enabled, failed: item.report && item.report.error && !item.report.incomplete }">
        <div class="step-head">
          <span class="step-number">{{ item.index + 1 }}</span>
          <button class="step-title" @click="open(item.index)" :aria-expanded="prep.openStep === item.index">
            <span class="strong">{{ item.spec ? item.spec.label : item.step.op }}</span>
            <span v-if="item.summary" class="subtle text-xs truncate">{{ item.summary }}</span>
          </button>
          <div class="step-actions">
            <button class="btn btn-ghost btn-icon" @click="moveStep(item.index, -1)" :disabled="item.index === 0" aria-label="Move up" title="Move up"><Icon name="arrow_upward" /></button>
            <button class="btn btn-ghost btn-icon" @click="moveStep(item.index, 1)" :disabled="item.index === prep.steps.length - 1" aria-label="Move down" title="Move down"><Icon name="arrow_downward" /></button>
            <button class="btn btn-ghost btn-icon" @click="toggleStep(item.index)" :aria-label="item.step.enabled ? 'Turn off' : 'Turn on'"
                    :title="item.step.enabled ? 'Turn this step off' : 'Turn this step on'"><Icon :name="item.step.enabled ? 'toggle_on' : 'toggle_off'" /></button>
            <button class="btn btn-ghost btn-icon" @click="removeStep(item.index)" aria-label="Remove step" title="Remove"><Icon name="delete_outline" /></button>
          </div>
        </div>
        <div v-if="item.report" class="step-report text-xs" :class="item.report.incomplete ? 'is-incomplete' : (item.report.error ? 'is-error' : '')">
          <template v-if="!item.step.enabled">Turned off</template>
          <template v-else-if="item.report.incomplete"><Icon name="edit_note" />Not run yet: {{ item.report.error }}</template>
          <template v-else-if="item.report.error"><Icon name="error_outline" />{{ item.report.error }}</template>
          <template v-else>{{ item.report.note }}<span v-if="item.report.rows_out !== item.report.rows_in" class="subtle">
            · {{ formatNumber(item.report.rows_in) }} → {{ formatNumber(item.report.rows_out) }} rows</span></template>
        </div>
        <div v-if="prep.openStep === item.index" class="step-body"><StepForm :index="item.index" /></div>
      </li>
    </ol>`,
};

export const PreviewTable = {
  props: { result: Object },
  setup(props) {
    const profiles = computed(() => Object.fromEntries(((props.result && props.result.profile) || []).map(p => [p.name, p])));
    return { profiles, truncate, KIND_ICONS };
  },
  template: /* html */ `
    <div class="preview-scroll">
      <table class="table table-compact preview-table">
        <thead>
          <tr>
            <th v-for="c in result.columns" :key="c">
              <div class="truncate" :title="c">{{ c }}</div>
              <div v-if="profiles[c]" class="column-meta">
                <span class="kind" :title="profiles[c].dtype"><Icon :name="KIND_ICONS[profiles[c].kind] || 'notes'" />{{ profiles[c].kind }}</span>
                <span v-if="profiles[c].missing" class="missing">{{ profiles[c].missing_pct }}% empty</span>
              </div>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, i) in result.head" :key="i">
            <td v-for="c in result.columns" :key="c" :title="row[c] === null ? '' : String(row[c])"
                :class="{ empty: row[c] === null }">{{ row[c] === null ? '—' : truncate(String(row[c]), 40) }}</td>
          </tr>
        </tbody>
      </table>
      <p v-if="!result.head.length" class="empty-note muted text-sm">No rows left after these steps.</p>
    </div>`,
};

export const ProfileTable = {
  props: { profile: Array },
  setup() {
    function summary(p) {
      if (p.kind === 'number' && p.min !== undefined) {
        return 'min ' + formatNumber(p.min) + ' · max ' + formatNumber(p.max) + ' · mean ' + formatNumber(p.mean);
      }
      if (p.kind === 'date' && p.min) return p.min.slice(0, 10) + ' → ' + p.max.slice(0, 10);
      if (p.top) return p.top.map(t => t.value + ' (' + t.count + ')').join(', ');
      return '';
    }
    return { summary, formatNumber, KIND_ICONS };
  },
  template: /* html */ `
    <div class="table-wrap flush">
      <table class="table table-compact">
        <thead><tr><th>Column</th><th>Kind</th><th class="right">Empty</th><th class="right">Distinct</th><th>Values</th></tr></thead>
        <tbody>
          <tr v-for="p in profile" :key="p.name">
            <td class="strong">{{ p.name }}</td>
            <td><span class="kind"><Icon :name="KIND_ICONS[p.kind] || 'notes'" />{{ p.kind }}</span></td>
            <td class="right num" :class="{ 'missing': p.missing }">{{ p.missing ? formatNumber(p.missing) + ' (' + p.missing_pct + '%)' : '—' }}</td>
            <td class="right num">{{ formatNumber(p.unique) }}</td>
            <td class="subtle text-xs truncate" :title="summary(p)">{{ summary(p) }}</td>
          </tr>
        </tbody>
      </table>
    </div>`,
};

export const AddStepDialog = {
  props: { open: Boolean },
  emits: ['close'],
  setup(props, { emit }) {
    const groups = computed(() => prep.categories.map(category => ({
      category, steps: prep.specs.filter(s => s.category === category),
    })));
    function choose(op) { addStep(op); emit('close'); }
    return { groups, choose };
  },
  template: /* html */ `
    <ModalDialog :open="open" title="Add a step" @close="$emit('close')">
      <div class="stack">
        <div v-for="group in groups" :key="group.category" class="stack-xs">
          <div class="field-label">{{ group.category }}</div>
          <div class="step-choices">
            <button v-for="s in group.steps" :key="s.op" class="check" @click="choose(s.op)">
              <div><div class="check-title">{{ s.label }}</div><div class="check-hint">{{ s.help }}</div></div>
            </button>
          </div>
        </div>
      </div>
      <template #footer><button class="btn btn-secondary" @click="$emit('close')">Cancel</button></template>
    </ModalDialog>`,
};

export const RecipesDialog = {
  props: { open: Boolean },
  emits: ['close'],
  setup(props, { emit }) {
    watch(() => props.open, isOpen => { if (isOpen) loadRecipes(); });
    async function use(recipe) { await useRecipe(recipe.id); emit('close'); }
    return { prep, use, deleteRecipe };
  },
  template: /* html */ `
    <ModalDialog :open="open" title="Saved recipes" @close="$emit('close')">
      <EmptyState v-if="!prep.recipes.length" icon="receipt_long" title="No recipes yet"
                  text="Build some steps and choose Save recipe to reuse them on new files." />
      <div v-else class="stack-xs">
        <div v-for="r in prep.recipes" :key="r.id" class="recipe-row">
          <div class="truncate">
            <div class="strong truncate">{{ r.name }}</div>
            <div class="subtle text-xs">{{ r.steps }} step{{ r.steps === 1 ? '' : 's' }} · {{ (r.updated_at || '').replace('T', ' ') }}</div>
          </div>
          <div class="row-nowrap">
            <button class="btn btn-secondary btn-sm" @click="use(r)">Use</button>
            <button class="btn btn-ghost btn-icon" @click="deleteRecipe(r)" :aria-label="'Delete ' + r.name"><Icon name="delete_outline" /></button>
          </div>
        </div>
        <p class="subtle text-xs">Use a recipe from Python with <code>easyml.prep.apply("name", "file.csv")</code>
          or from the command line with <code>easy-ml prep apply "name" file.csv -o clean.csv</code>.</p>
      </div>
      <template #footer><button class="btn btn-secondary" @click="$emit('close')">Close</button></template>
    </ModalDialog>`,
};

export const SaveRecipeDialog = {
  props: { open: Boolean },
  emits: ['close'],
  setup(props, { emit }) {
    const name = ref('');
    const asNew = ref(false);
    const saving = ref(false);
    watch(() => props.open, isOpen => {
      if (!isOpen) return;
      asNew.value = false;
      name.value = prep.recipe ? prep.recipe.name : cleanFileName().replace(/_clean\.csv$/, '');
    });
    async function save() {
      saving.value = true;
      const ok = await saveRecipe(name.value.trim(), asNew.value || !prep.recipe);
      saving.value = false;
      if (ok) emit('close');
    }
    return { prep, name, asNew, saving, save };
  },
  template: /* html */ `
    <ModalDialog :open="open" title="Save recipe" @close="$emit('close')">
      <div class="stack">
        <p class="muted text-sm">A recipe stores these {{ prep.steps.length }} steps so you can apply them to new files later:
          here, when retraining a run, from Python or from the command line.</p>
        <div class="field">
          <label class="field-label" for="recipe-name">Name</label>
          <input id="recipe-name" class="input" v-model="name" maxlength="80" />
        </div>
        <label v-if="prep.recipe" class="row text-sm"><input type="checkbox" v-model="asNew" />
          <span>Save as a new recipe (instead of updating "{{ prep.recipe.name }}")</span></label>
      </div>
      <template #footer>
        <button class="btn btn-secondary" @click="$emit('close')">Cancel</button>
        <button class="btn btn-primary" :disabled="!name.trim() || saving" @click="save"><Icon name="save" />Save</button>
      </template>
    </ModalDialog>`,
};

const TRAIN_TASKS = [
  { id: 'MLClassification', label: 'Predict a category' },
  { id: 'MLRegression', label: 'Predict a number' },
  { id: 'NLPTextClassification', label: 'Classify text' },
];

export const TrainDialog = {
  props: { open: Boolean },
  emits: ['close'],
  setup(props, { emit }) {
    const task = ref('MLClassification');
    const target = ref('');
    const textColumn = ref('');
    const columns = computed(() => (prep.result ? prep.result.columns : []));
    watch(() => props.open, isOpen => { if (isOpen && !columns.value.includes(target.value)) target.value = ''; });
    const ready = computed(() => target.value && (task.value !== 'NLPTextClassification' || (textColumn.value && textColumn.value !== target.value)));
    async function start() {
      await trainWithData(task.value, { target_column: target.value, text_column: task.value === 'NLPTextClassification' ? textColumn.value : '' });
      emit('close');
    }
    return { prep, task, target, textColumn, columns, ready, start, TRAIN_TASKS, featureAvailable };
  },
  template: /* html */ `
    <ModalDialog :open="open" title="Train with this data" @close="$emit('close')">
      <div class="stack">
        <p class="muted text-sm">Applies the steps to the whole file, then opens a new run with the cleaned data.
          The run records these steps, so retraining it later cleans new files the same way.</p>
        <div class="segmented" role="group" aria-label="What to predict">
          <button v-for="t in TRAIN_TASKS" :key="t.id" :class="{ active: task === t.id }" @click="task = t.id">{{ t.label }}</button>
        </div>
        <div class="field">
          <label class="field-label" for="train-target">Column to predict</label>
          <select id="train-target" class="select" v-model="target">
            <option value="">Choose…</option>
            <option v-for="c in columns" :key="c" :value="c">{{ c }}</option>
          </select>
        </div>
        <div v-if="task === 'NLPTextClassification'" class="field">
          <label class="field-label" for="train-text">Text column</label>
          <select id="train-text" class="select" v-model="textColumn">
            <option value="">Choose…</option>
            <option v-for="c in columns" :key="c" :value="c">{{ c }}</option>
          </select>
        </div>
      </div>
      <template #footer>
        <button class="btn btn-secondary" @click="$emit('close')">Cancel</button>
        <button class="btn btn-primary" :disabled="!ready || prep.busy === 'apply'" @click="start">
          <Spinner v-if="prep.busy === 'apply'" /><Icon v-else name="model_training" />Continue</button>
      </template>
    </ModalDialog>`,
};
