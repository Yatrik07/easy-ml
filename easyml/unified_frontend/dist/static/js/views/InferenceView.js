import { computed } from '../lib/vue.js';
import { formatPercent, formatPrediction, formatScore } from '../lib/format.js';
import { featureAvailable, modelLabel, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import {
  columnChoices, explain, exportModel, inference, predict, predictBatch, setImage,
} from '../store/inference.js';

function pyValue(value, numeric) {
  if (value === '' || value === null || value === undefined) return 'None';
  return numeric && !Number.isNaN(Number(value)) ? String(value) : JSON.stringify(String(value));
}

export default {
  setup() {
    const isTabular = computed(() => ['MLClassification', 'MLRegression'].includes(inference.task));
    const isText = computed(() => inference.task === 'NLPTextClassification');
    const isImage = computed(() => inference.task === 'ImageClassification');
    const usesBest = computed(() => inference.modelName === inference.bestModel);

    const codeTabs = computed(() => {
      const id = inference.runId;
      const modelArg = usesBest.value ? '' : ', model_name="' + inference.modelName + '"';
      const cliModel = usesBest.value ? '' : ' --model ' + inference.modelName;
      let example;
      let cli;
      if (isText.value) {
        example = 'model.predict(["Your text to classify"])';
        cli = 'easy-ml predict ' + id + ' "Your text to classify"' + cliModel + '\neasy-ml predict ' + id + ' texts.csv --output predictions.csv' + cliModel;
      } else if (isImage.value) {
        example = 'model.predict("path/to/image.jpg")   # or a list of paths, or a folder';
        cli = 'easy-ml predict ' + id + ' path/to/image.jpg' + cliModel + '\neasy-ml predict ' + id + ' path/to/folder --output predictions.csv' + cliModel;
      } else {
        const cols = inference.columns.slice(0, 4).map(col => {
          const numeric = !columnChoices(col).length;
          return '    "' + col + '": ' + pyValue(inference.values[col], numeric) + ',';
        });
        const extra = inference.columns.length - 4;
        if (extra > 0) cols.push('    # ... ' + extra + ' more column' + (extra === 1 ? '' : 's'));
        example = 'model.predict({\n' + cols.join('\n') + '\n})\n\n# or a whole file / DataFrame:\nmodel.predict("new_data.csv")';
        cli = 'easy-ml predict ' + id + ' new_data.csv --output predictions.csv' + cliModel;
      }
      return [
        { id: 'python', label: 'Python', code: 'import easyml\n\nmodel = easyml.load_model(' + id + modelArg + ')\n' + example },
        { id: 'cli', label: 'Command line', code: 'easy-ml runs            # list your runs\n' + cli },
      ];
    });

    function onBatch(event) { inference.batchFile = event.target.files[0] || null; }
    function onImage(event) { setImage(event.target.files[0]); }

    return {
      inference, isTabular, isText, isImage, codeTabs, columnChoices, predict, predictBatch, explain, exportModel,
      onBatch, onImage, featureAvailable, modelLabel, taskLabel, navigate, formatPrediction, formatPercent, formatScore,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader :eyebrow="taskLabel(inference.task) + ' · run #' + inference.runId" :title="'Predict with ' + (inference.runName || 'run #' + inference.runId)"
                  :subtitle="inference.datasetName ? 'Trained on ' + inference.datasetName : ''">
        <button class="btn btn-secondary" @click="navigate('history')"><Icon name="history" />All runs</button>
      </PageHeader>

      <div class="inference-layout">
        <div class="stack">
          <section class="card">
            <div class="card-body stack">
              <div class="field">
                <label for="model-select">Model</label>
                <select id="model-select" class="select" v-model="inference.modelName">
                  <option v-for="m in inference.models" :key="m" :value="m">
                    {{ modelLabel(m) }}{{ m === inference.bestModel ? '  ★ best' : '' }}{{ inference.modelScores[m] != null ? '  (' + formatScore(inference.modelScores[m], inference.metric) + ')' : '' }}
                  </option>
                </select>
              </div>

              <template v-if="isTabular">
                <div class="segmented" role="group" aria-label="Input mode">
                  <button :class="{ active: inference.mode === 'manual' }" @click="inference.mode = 'manual'"><Icon name="edit_note" size="sm" />Single row</button>
                  <button :class="{ active: inference.mode === 'batch' }" @click="inference.mode = 'batch'"><Icon name="upload_file" size="sm" />CSV file</button>
                </div>
                <div v-if="inference.mode === 'manual'" class="form-grid">
                  <div v-for="col in inference.columns" :key="col" class="field">
                    <label :for="'in-' + col" class="truncate" :title="col">{{ col }}</label>
                    <select v-if="columnChoices(col).length" :id="'in-' + col" class="select" v-model="inference.values[col]">
                      <option value="">(missing)</option>
                      <option v-for="v in columnChoices(col)" :key="String(v)" :value="v">{{ v }}</option>
                    </select>
                    <input v-else :id="'in-' + col" class="input" type="number" step="any" v-model="inference.values[col]" placeholder="number" />
                  </div>
                </div>
                <div v-else class="field">
                  <label for="batch-file">CSV with the same columns</label>
                  <input id="batch-file" class="input" type="file" accept=".csv" @change="onBatch" />
                  <span class="field-hint">You get the file back with a Prediction column added.</span>
                </div>
                <p class="field-hint" v-if="inference.mode === 'manual'">Leave a field empty and it is filled in the same way missing values were during training.</p>
              </template>

              <div v-if="isText" class="field">
                <label for="text-input">Text</label>
                <textarea id="text-input" class="textarea" v-model="inference.text" placeholder="Paste or type the text to classify"></textarea>
              </div>

              <div v-if="isImage" class="field">
                <label for="image-input">Image</label>
                <input id="image-input" class="input" type="file" accept="image/*" @change="onImage" />
                <img v-if="inference.imagePreview" :src="inference.imagePreview" class="image-preview" alt="Selected image" />
              </div>

              <div class="row">
                <button v-if="!isTabular || inference.mode === 'manual'" class="btn btn-primary" @click="predict" :disabled="inference.predicting">
                  <Spinner v-if="inference.predicting" /><Icon v-else name="bolt" />Predict
                </button>
                <button v-else class="btn btn-primary" @click="predictBatch" :disabled="inference.predicting || !inference.batchFile">
                  <Spinner v-if="inference.predicting" /><Icon v-else name="download" />Predict and download
                </button>
                <button v-if="isTabular" class="btn btn-secondary" @click="exportModel" :disabled="inference.exporting"
                        title="Download a FastAPI service with Dockerfile for this model">
                  <Spinner v-if="inference.exporting" /><Icon v-else name="deployed_code" />Export as API
                </button>
              </div>
            </div>
          </section>

          <section class="card">
            <div class="card-header"><div><h3>Use this model in code</h3>
              <div class="card-hint">Run on this computer, with the same Easy-ML data directory (set EASYML_DATA_DIR if you changed it).</div></div></div>
            <div class="card-body"><CodeTabs :key="inference.runId + inference.modelName" :tabs="codeTabs" /></div>
          </section>
        </div>

        <aside class="card inference-result">
          <div class="card-header"><h3>Prediction</h3></div>
          <div class="card-body stack">
            <template v-if="inference.result">
              <div class="prediction-value">{{ formatPrediction(inference.result.prediction) }}</div>
              <div v-if="inference.result.probability != null" class="stack-sm">
                <div class="row-between text-sm"><span class="muted">Confidence</span><span class="strong num">{{ formatPercent(inference.result.probability) }}</span></div>
                <div class="progress"><div class="progress-bar" :style="{ width: (inference.result.probability * 100) + '%' }"></div></div>
              </div>
              <button v-if="isTabular && featureAvailable('explain')" class="btn btn-secondary" @click="explain" :disabled="inference.explaining">
                <Spinner v-if="inference.explaining" /><Icon v-else name="insights" />Explain this prediction
              </button>
            </template>
            <p v-else class="muted text-sm">Fill in the inputs and press Predict.</p>
            <div v-if="inference.shapPlot" class="stack-sm">
              <div class="field-label">Why this prediction (SHAP)</div>
              <div class="figure"><img :src="'data:image/png;base64,' + inference.shapPlot" alt="SHAP waterfall plot" /></div>
              <p class="field-hint">Red bars push the prediction up, blue bars push it down.</p>
            </div>
          </div>
        </aside>
      </div>
    </div>`,
};
