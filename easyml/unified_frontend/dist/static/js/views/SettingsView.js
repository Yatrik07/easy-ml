import { onMounted, ref } from '../lib/vue.js';
import { deleteJSON, getJSON } from '../lib/api.js';
import { formatNumber } from '../lib/format.js';
import { setThemePreference, themePreference } from '../lib/theme.js';
import { session } from '../store/session.js';
import { confirmAction, toastError, toastSuccess } from '../store/ui.js';
import { loadRuns } from '../store/runs.js';

const FEATURES = [
  { id: 'boost', label: 'XGBoost, LightGBM, CatBoost', extra: 'boost' },
  { id: 'cv', label: 'Image classification (PyTorch)', extra: 'cv' },
  { id: 'embeddings', label: 'Sentence embeddings for text', extra: 'nlp' },
  { id: 'lemmatize', label: 'Lemmatization (spaCy)', extra: 'nlp' },
  { id: 'explain', label: 'SHAP explanations', extra: 'eda' },
  { id: 'eda_report', label: 'Automatic EDA reports', extra: 'eda' },
];

export default {
  setup() {
    const info = ref(null);
    const device = ref(null);
    const loading = ref(false);
    const clearing = ref(false);

    async function refresh() {
      loading.value = true;
      try {
        [info.value, device.value] = await Promise.all([
          getJSON('/settings/info', 'Could not read storage information'),
          getJSON('/settings/device', 'Could not read device information'),
        ]);
      } catch (e) {
        toastError(e.message);
      } finally {
        loading.value = false;
      }
    }
    onMounted(refresh);

    function available(feature) {
      const caps = session.capabilities;
      if (!caps) return false;
      if (feature.id === 'boost') return (caps.models.MLClassification || []).includes('xg_boost');
      return !!caps.features[feature.id];
    }
    function hint(feature) {
      if (feature.id === 'lemmatize' && session.capabilities) return session.capabilities.install_hints.lemmatize;
      return 'pip install "easy-ml-studio[' + feature.extra + ']"';
    }

    async function clearData() {
      const ok = await confirmAction({
        title: 'Delete all data?',
        message: 'This permanently deletes every run, trained model, upload and log in the data directory. This cannot be undone.',
        confirmLabel: 'Delete everything',
        danger: true,
      });
      if (!ok) return;
      clearing.value = true;
      try {
        const res = await deleteJSON('/settings/clear-data', {}, 'Could not clear data');
        toastSuccess(res.message || 'All data cleared.');
        await Promise.all([refresh(), loadRuns()]);
      } catch (e) {
        toastError(e.message);
      } finally {
        clearing.value = false;
      }
    }

    const themes = [
      { id: 'system', label: 'System', icon: 'computer' },
      { id: 'light', label: 'Light', icon: 'light_mode' },
      { id: 'dark', label: 'Dark', icon: 'dark_mode' },
    ];

    return {
      info, device, loading, clearing, refresh, clearData, FEATURES, available, hint,
      themes, themePreference, setThemePreference, formatNumber,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader title="Settings" subtitle="Appearance, storage and the optional features of this installation.">
        <button class="btn btn-secondary" @click="refresh" :disabled="loading"><Icon name="refresh" />Refresh</button>
      </PageHeader>

      <section class="card">
        <div class="card-header"><div><h3>Appearance</h3><div class="card-hint">System follows your operating system's light or dark mode.</div></div></div>
        <div class="card-body">
          <div class="segmented" role="group" aria-label="Theme">
            <button v-for="t in themes" :key="t.id" :class="{ active: themePreference === t.id }" @click="setThemePreference(t.id)">
              <Icon :name="t.icon" size="sm" />{{ t.label }}
            </button>
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-header"><div><h3>Storage</h3><div class="card-hint">Runs, uploads and logs live here, outside the package, so upgrades never delete them. Change it with the EASYML_DATA_DIR environment variable.</div></div></div>
        <div class="card-body stack">
          <div v-if="info" class="stack-sm">
            <div class="field-label">Data directory</div>
            <CopyField :text="info.data_dir" label="Copy path" />
          </div>
          <div v-if="info" class="grid-3">
            <StatTile label="Used by Easy-ML" :value="formatNumber(info.storage_used_mb, 1) + ' MB'" />
            <StatTile label="Free disk space" :value="formatNumber(info.disk_free_mb / 1024, 1) + ' GB'" />
            <StatTile label="Version" :value="info.version" />
          </div>
          <div v-else class="row muted"><Spinner /> Loading…</div>
        </div>
      </section>

      <section class="card">
        <div class="card-header"><div><h3>Compute</h3><div class="card-hint">Image models train much faster on an NVIDIA GPU.</div></div></div>
        <div class="card-body stack" v-if="device">
          <div class="row">
            <span class="badge" :class="device.cuda_available ? 'badge-success' : ''">
              <Icon :name="device.cuda_available ? 'bolt' : 'memory'" />{{ device.cuda_available ? 'GPU available' : 'CPU only' }}
            </span>
            <span class="muted text-sm">{{ device.cuda_device_name }}</span>
            <span class="subtle text-sm">PyTorch {{ device.torch_version }}</span>
          </div>
          <div v-if="device.torch_version !== 'Not installed' && !device.cuda_available" class="stack-sm">
            <p class="muted text-sm">Have an NVIDIA GPU? Reinstall PyTorch with CUDA support. The right command depends on your driver; see
              <a :href="device.help_url" target="_blank" rel="noopener noreferrer">pytorch.org</a>.</p>
            <CodeBlock :code="device.cuda_install_hint" />
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-header"><div><h3>Optional features</h3><div class="card-hint">Install extras to unlock more models and tools, then restart easy-ml.</div></div></div>
        <div class="table-wrap flush">
          <table class="table">
            <tbody>
              <tr v-for="f in FEATURES" :key="f.id">
                <td>{{ f.label }}</td>
                <td>
                  <span v-if="available(f)" class="badge badge-success"><Icon name="check" />Installed</span>
                  <span v-else class="badge">Not installed</span>
                </td>
                <td class="right"><code v-if="!available(f)" class="subtle text-xs">{{ hint(f) }}</code></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="card">
        <div class="card-header"><div><h3>Danger zone</h3><div class="card-hint">Remove every run, model, upload and log from the data directory.</div></div></div>
        <div class="card-body">
          <button class="btn btn-danger-outline" @click="clearData" :disabled="clearing"><Icon name="delete_forever" />Delete all data</button>
        </div>
      </section>
    </div>`,
};
