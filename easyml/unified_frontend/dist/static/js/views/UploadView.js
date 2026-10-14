import { computed, ref } from '../lib/vue.js';
import { formatNumber, truncate } from '../lib/format.js';
import { featureAvailable, installHint, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import {
  confirmImageFolder, dataReady, isImage, isText, openEdaReport, openFolderPicker, uploadDataset, wizard,
} from '../store/wizard.js';

export default {
  setup() {
    const fileInput = ref(null);
    const dragging = ref(false);
    const upload = computed(() => wizard.upload);
    const columns = computed(() => (wizard.upload.data ? wizard.upload.data.column_names : []));
    const preview = computed(() => (wizard.upload.data ? wizard.upload.data.data_head || [] : []));

    function pick() { if (fileInput.value) fileInput.value.click(); }
    function onFile(event) {
      uploadDataset(event.target.files[0]);
      event.target.value = '';
    }
    function onDrop(event) {
      dragging.value = false;
      uploadDataset(event.dataTransfer.files[0]);
    }

    const subtitle = computed(() => (isImage.value
      ? 'Choose a folder of images with one sub-folder per class, for example cats/ and dogs/.'
      : 'Upload a CSV or Excel file. The first row must contain the column names.'));

    return {
      wizard, upload, columns, preview, fileInput, dragging, pick, onFile, onDrop, isImage, isText, dataReady,
      subtitle, taskLabel, navigate, openEdaReport, openFolderPicker, confirmImageFolder, featureAvailable,
      installHint, formatNumber, truncate,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader :eyebrow="'Step 1 · ' + taskLabel(wizard.config.task)" :title="isImage ? 'Choose your images' : 'Upload your dataset'" :subtitle="subtitle" />

      <!-- Tabular and text datasets -->
      <template v-if="!isImage">
        <div v-if="!upload.data" class="dropzone" :class="{ dragging }" role="button" tabindex="0"
             @click="pick" @keydown.enter="pick" @dragover.prevent="dragging = true" @dragleave="dragging = false" @drop.prevent="onDrop">
          <input ref="fileInput" type="file" accept=".csv,.xlsx,.xls" hidden @change="onFile" />
          <template v-if="upload.uploading"><Spinner /><strong>Uploading {{ upload.fileName }}…</strong></template>
          <template v-else>
            <Icon name="upload_file" />
            <strong>Drop a file here or click to browse</strong>
            <span class="text-sm">CSV, XLSX or XLS</span>
          </template>
        </div>

        <template v-else>
          <div class="card">
            <div class="card-header">
              <div class="row">
                <Icon name="description" />
                <div>
                  <h3>{{ upload.fileName }}</h3>
                  <div class="card-hint">{{ formatNumber(upload.data.rows) }} rows · {{ upload.data.cols }} columns · {{ formatNumber(upload.data.dataset_size_kb, 1) }} KB</div>
                </div>
              </div>
              <div class="row">
                <button class="btn btn-secondary btn-sm" @click="openEdaReport" :disabled="wizard.edaLoading || !featureAvailable('eda_report')"
                        :title="featureAvailable('eda_report') ? 'Open an automatic exploratory data analysis report' : 'Needs: ' + installHint('eda_report')">
                  <Spinner v-if="wizard.edaLoading" /><Icon v-else name="query_stats" />Explore data
                </button>
                <button class="btn btn-ghost btn-sm" @click="pick"><Icon name="swap_horiz" />Replace file</button>
                <input ref="fileInput" type="file" accept=".csv,.xlsx,.xls" hidden @change="onFile" />
              </div>
            </div>
            <div class="card-body form-grid">
              <div v-if="isText" class="field">
                <label for="text-column">Text column</label>
                <select id="text-column" class="select" v-model="wizard.config.text_column">
                  <option value="" disabled>Choose the column with the text</option>
                  <option v-for="col in columns" :key="col" :value="col">{{ col }}</option>
                </select>
              </div>
              <div class="field">
                <label for="target-column">Column to predict</label>
                <select id="target-column" class="select" v-model="wizard.config.target_column">
                  <option value="" disabled>Choose the target column</option>
                  <option v-for="col in columns" :key="col" :value="col">{{ col }}</option>
                </select>
              </div>
              <div class="field">
                <label for="run-name">Run name</label>
                <input id="run-name" class="input" v-model="wizard.config.name" placeholder="e.g. churn_v1" />
                <span v-if="!wizard.config.name.trim()" class="field-error">Give this run a name.</span>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-header"><div><h3>Preview</h3><div class="card-hint">First rows of the file.</div></div></div>
            <div class="table-wrap flush">
              <table class="table table-compact preview-table">
                <thead><tr><th v-for="col in columns" :key="col">
                  {{ col }}
                  <span v-if="col === wizard.config.target_column" class="badge badge-accent">Target</span>
                  <span v-if="col === wizard.config.text_column" class="badge">Text</span>
                </th></tr></thead>
                <tbody>
                  <tr v-for="(row, i) in preview" :key="i">
                    <td v-for="col in columns" :key="col" :title="row[col]">{{ row[col] === null ? '—' : truncate(row[col], 40) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </template>
      </template>

      <!-- Image datasets -->
      <div v-else class="card">
        <div class="card-body stack">
          <div class="field">
            <label for="image-folder">Image folder</label>
            <div class="row row-nowrap">
              <input id="image-folder" class="input" v-model="wizard.config.dataset_path" @input="wizard.cvConfirmed = false"
                     placeholder="C:\\data\\animals  (folders: cats\\, dogs\\, …)" />
              <button class="btn btn-secondary" @click="openFolderPicker"><Icon name="folder_open" />Browse</button>
            </div>
            <span class="field-hint">One folder that contains a sub-folder per class, or several class folders separated by commas.</span>
          </div>
          <div class="field">
            <label for="cv-run-name">Run name</label>
            <input id="cv-run-name" class="input" v-model="wizard.config.name" placeholder="e.g. animals_v1" />
          </div>
          <div class="row">
            <button class="btn btn-secondary" @click="confirmImageFolder" :disabled="!wizard.config.dataset_path.trim()">
              <Icon name="check" />Use this folder
            </button>
            <span v-if="wizard.cvConfirmed" class="badge badge-success"><Icon name="check" />Folder selected
              <template v-if="wizard.cvClassCount"> · {{ wizard.cvClassCount }} classes</template></span>
          </div>
        </div>
      </div>

      <div class="row-between">
        <button class="btn btn-ghost" @click="navigate('home')"><Icon name="arrow_back" />Back</button>
        <button class="btn btn-primary" :disabled="!dataReady" @click="navigate('config')">Continue<Icon name="arrow_forward" /></button>
      </div>

      <FolderPicker />
    </div>`,
};
