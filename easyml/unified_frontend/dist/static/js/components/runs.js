// Run History extras: retrain dialog, run import dialog and the "retrained from" badge.
import { computed, ref } from '../lib/vue.js';
import { postForm } from '../lib/api.js';
import { cleaningRecipe, closeRetrain, columnCheck, retrain, submitRetrain, uploadRetrainFile } from '../store/retrain.js';
import { loadRuns, openRunDetail, runs } from '../store/runs.js';
import { training } from '../store/training.js';
import { toast, toastError, toastSuccess } from '../store/ui.js';

export const LineageBadge = {
  props: { run: Object },
  setup(props) {
    const parentExists = computed(() => runs.list.some(r => r.id === props.run.parent_run_id));
    return { parentExists, openRunDetail };
  },
  template: /* html */ `
    <button v-if="run && run.parent_run_id" class="badge badge-accent lineage" @click.stop="openRunDetail(run.parent_run_id)"
            :disabled="!parentExists" :title="parentExists ? 'Open run #' + run.parent_run_id : 'Run #' + run.parent_run_id + ' was deleted'">
      <Icon :name="run.origin === 'schedule' ? 'event_available' : 'autorenew'" />{{ run.origin === 'schedule' ? 'Scheduled retrain of' : 'Retrained from' }} #{{ run.parent_run_id }}<span v-if="!parentExists">&nbsp;(deleted)</span>
    </button>
    <span v-else-if="run && run.origin === 'import'" class="badge lineage" title="Imported from a run archive"><Icon name="move_to_inbox" />Imported</span>`,
};

export const RetrainDialog = {
  setup() {
    const check = computed(() => columnCheck());
    const spec = computed(() => (retrain.spec && retrain.spec.retrain) || null);
    // With a recipe the raw file's columns can differ (renamed, combined); the server checks them after cleaning.
    const cleaning = computed(() => cleaningRecipe());
    const blocking = computed(() => (cleaning.value ? [] : check.value.missing));
    const isFolder = computed(() => !!spec.value && spec.value.input === 'image_folder');
    const busy = computed(() => training.status === 'running');
    const ready = computed(() => {
      if (!spec.value || !spec.value.supported || busy.value || !retrain.name.trim()) return false;
      if (isFolder.value) return !!retrain.folder.trim();
      return !!retrain.upload.savedPath && !blocking.value.length;
    });
    function onFile(event) { uploadRetrainFile(event.target.files[0]); }
    return { retrain, spec, check, cleaning, blocking, isFolder, busy, ready, onFile, closeRetrain, submitRetrain };
  },
  template: /* html */ `
    <ModalDialog :open="retrain.open" :title="'Retrain run #' + retrain.runId" @close="closeRetrain">
      <div v-if="retrain.loading" class="row muted"><Spinner /> Loading run…</div>
      <div v-else-if="retrain.error" class="alert alert-danger"><Icon name="error_outline" /><div>{{ retrain.error }}</div></div>
      <div v-else-if="spec" class="stack">
        <p class="muted text-sm">Trains <strong>{{ retrain.spec.run_name }}</strong> again on new data with exactly the same settings,
          as a new run. The original run is kept.</p>
        <div v-if="!spec.supported" class="alert alert-warning"><Icon name="warning_amber" /><div>{{ spec.reason }}</div></div>
        <template v-else>
          <div v-if="!isFolder" class="stack-sm">
            <div class="field-label">New data</div>
            <div class="subtle text-xs">{{ cleaning ? 'After cleaning, the data needs these columns:' : 'Needs these columns:' }}</div>
            <div class="term-list"><span v-for="c in spec.required_columns" :key="c" class="badge"
                 :class="{ 'badge-danger': blocking.includes(c) }">{{ c }}</span></div>
            <label class="btn btn-secondary file-button">
              <Icon name="upload_file" />{{ retrain.upload.fileName || 'Choose a CSV or Excel file' }}
              <input type="file" accept=".csv,.xlsx,.xls" class="sr-only" @change="onFile" />
            </label>
            <div v-if="retrain.upload.uploading" class="row muted text-sm"><Spinner /> Uploading…</div>
            <div v-if="blocking.length" class="alert alert-danger"><Icon name="error_outline" />
              <div>This file is missing {{ blocking.length }} column(s) the run was trained on: {{ blocking.join(', ') }}.
                <span v-if="retrain.recipes.length">If a recipe makes them, choose it below.</span></div></div>
            <div v-else-if="retrain.upload.savedPath && cleaning" class="alert alert-info"><Icon name="cleaning_services" />
              <div>{{ retrain.upload.rows.toLocaleString() }} rows. They are cleaned with recipe "{{ cleaning.name }}" first;
                the columns are checked after cleaning.</div></div>
            <div v-else-if="retrain.upload.savedPath" class="alert alert-success"><Icon name="check_circle" />
              <div>{{ retrain.upload.rows.toLocaleString() }} rows with every column the run needs.
                <span v-if="check.extra.length">{{ check.extra.length }} other column(s) will be ignored.</span></div></div>
          </div>
          <div v-else class="stack-sm">
            <div class="field-label">New image folder</div>
            <div v-if="retrain.folders.length" class="stack-sm" role="listbox">
              <label v-for="f in retrain.folders" :key="f.path" class="check" :class="{ checked: retrain.folder === f.path }">
                <input type="radio" name="retrain-folder" :value="f.path" v-model="retrain.folder" />
                <div><div class="check-title">{{ f.name }}</div>
                  <div class="check-hint">{{ f.num_classes }} class folder{{ f.num_classes === 1 ? '' : 's' }}</div></div>
              </label>
            </div>
            <input class="input" v-model="retrain.folder" placeholder="Or type a folder path" aria-label="Image folder path" />
            <div v-if="spec.assumed_defaults.length" class="alert alert-warning"><Icon name="info" />
              <div>This run was trained before Easy-ML 0.6.0, so these settings use defaults: {{ spec.assumed_defaults.join(', ') }}.</div></div>
          </div>
          <div v-if="!isFolder && (spec.recipe || retrain.recipes.length)" class="stack-xs">
            <label class="field-label" for="retrain-recipe">Clean the new data first</label>
            <select id="retrain-recipe" class="select" v-model="retrain.recipeId">
              <option v-if="spec.recipe" value="">The run's recipe: {{ spec.recipe.name }} ({{ spec.recipe.steps }} steps)</option>
              <option :value="spec.recipe ? 'none' : ''">No cleaning, use the file as it is</option>
              <option v-for="r in retrain.recipes" :key="r.id" :value="String(r.id)">Recipe: {{ r.name }} ({{ r.steps }} steps)</option>
            </select>
            <div v-if="spec.recipe && retrain.recipeId === ''" class="subtle text-xs">This run was trained on cleaned data,
              so new files are cleaned with the same steps. A file that is already clean is used as it is.</div>
          </div>
          <div class="stack-xs">
            <label class="field-label" for="retrain-name">Name of the new run</label>
            <input id="retrain-name" class="input" v-model="retrain.name" maxlength="80" />
          </div>
          <div class="subtle text-xs">Models: {{ spec.models.filter(m => m.available).map(m => m.label).join(', ') }}</div>
          <div v-if="busy" class="alert alert-warning"><Icon name="hourglass_top" /><div>Another run is training. Start this one when it finishes.</div></div>
        </template>
      </div>
      <template #footer>
        <button class="btn btn-secondary" @click="closeRetrain">Cancel</button>
        <button class="btn btn-primary" :disabled="!ready" @click="submitRetrain"><Icon name="autorenew" />Start retraining</button>
      </template>
    </ModalDialog>`,
};

export const ImportRunDialog = {
  props: { open: Boolean },
  emits: ['close'],
  setup(props, { emit }) {
    const file = ref(null);
    const name = ref('');
    const trusted = ref(false);
    const busy = ref(false);
    function onFile(event) { file.value = event.target.files[0] || null; }
    function close() {
      file.value = null; name.value = ''; trusted.value = false;
      emit('close');
    }
    async function submit() {
      const form = new FormData();
      form.append('file', file.value);
      form.append('trusted', trusted.value ? 'true' : 'false');
      if (name.value.trim()) form.append('name', name.value.trim());
      busy.value = true;
      try {
        const result = await postForm('/runs/import', form, 'Import failed');
        toastSuccess('Imported as run #' + result.run_id + '.');
        (result.warnings || []).forEach(w => toast(w));
        close();
        await loadRuns();
        openRunDetail(result.run_id);
      } catch (e) {
        toastError(e.message);
      } finally {
        busy.value = false;
      }
    }
    return { file, name, trusted, busy, onFile, close, submit };
  },
  template: /* html */ `
    <ModalDialog :open="open" title="Import a run" @close="close">
      <div class="stack">
        <p class="muted text-sm">Load a run saved with “Download run (.zip)” or <code>easy-ml export</code> on another computer.
          It is added as a new run and keeps its models, scores and charts.</p>
        <label class="btn btn-secondary file-button">
          <Icon name="folder_zip" />{{ file ? file.name : 'Choose a .zip file' }}
          <input type="file" accept=".zip" class="sr-only" @change="onFile" />
        </label>
        <div class="stack-xs">
          <label class="field-label" for="import-name">Name (optional)</label>
          <input id="import-name" class="input" v-model="name" maxlength="80" placeholder="Keep the original name" />
        </div>
        <label class="check" :class="{ checked: trusted }">
          <input type="checkbox" v-model="trusted" />
          <div>
            <div class="check-title">I trust where this file came from</div>
            <div class="check-hint">Model files can run code on this computer when they are loaded. Only import runs from people you trust.</div>
          </div>
        </label>
      </div>
      <template #footer>
        <button class="btn btn-secondary" @click="close">Cancel</button>
        <button class="btn btn-primary" :disabled="!file || !trusted || busy" @click="submit">
          <Spinner v-if="busy" /><Icon v-else name="move_to_inbox" />Import run</button>
      </template>
    </ModalDialog>`,
};
