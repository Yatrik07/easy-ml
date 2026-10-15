import { computed, ref } from '../lib/vue.js';
import { formatPercent, formatPrediction, formatScore } from '../lib/format.js';
import { featureAvailable, modelLabel, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import {
  columnChoices, explain, exportModel, inference, predict, predictBatch, setImage,
} from '../store/inference.js';

// A value for the API example: what the user typed, else a real value from the training data.
function exampleValue(column) {
  const typed = inference.values[column];
  const choices = columnChoices(column);
  if (typed !== '' && typed !== null && typed !== undefined) {
    return choices.length || Number.isNaN(Number(typed)) ? String(typed) : Number(typed);
  }
  if (column in inference.exampleRow) return inference.exampleRow[column];
  return choices.length ? choices[0] : 0;
}

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

    // `easy-ml serve` plus one request in the format the service expects for this task.
    function serveExample() {
      const id = inference.runId;
      const cliModel = usesBest.value ? '' : ' --model ' + inference.modelName;
      let call;
      if (isImage.value) {
        call = 'curl -X POST http://127.0.0.1:8000/predict -F "file=@photo.jpg"';
      } else {
        let body;
        if (isText.value) {
          body = { text: 'Your text to classify' };
        } else {
          const row = {};
          for (const col of inference.columns) row[col] = exampleValue(col);
          body = { features: row };
        }
        call = 'curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d \'' + JSON.stringify(body) + '\'';
      }
      return [
        '# a REST API for this model, on this computer (Ctrl+C stops it)',
        'easy-ml serve ' + id + cliModel,
        '',
        '# from another terminal or program:',
        call,
        '',
        '# help, inputs and more examples: http://127.0.0.1:8000/',
      ].join('\n');
    }

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
        { id: 'serve', label: 'Serve as API', code: serveExample() },
      ];
    });

    const exportOpen = ref(false);
    const apiTabs = computed(() => {
      const row = {};
      for (const col of inference.columns) row[col] = exampleValue(col);
      const oneLine = JSON.stringify({ features: row });
      return [
        { id: 'run', label: 'Start it', code: [
          '# unzip the file, open a terminal in the folder, then:',
          'python -m venv .venv',
          '.venv\\Scripts\\activate          # macOS/Linux: source .venv/bin/activate',
          'pip install -r requirements.txt',
          'uvicorn main:app --port 8000',
          '',
          '# then open http://127.0.0.1:8000/docs to try it in the browser',
        ].join('\n') },
        { id: 'python', label: 'Python', code: [
          'import requests  # pip install requests',
          '',
          'row = ' + JSON.stringify(row, null, 4),
          'answer = requests.post("http://127.0.0.1:8000/predict", json={"features": row}).json()',
          'print(answer)   # ' + (inference.task === 'MLRegression' ? '{"prediction": 123.4}' : '{"prediction": ..., "confidence": 0.93}'),
        ].join('\n') },
        { id: 'powershell', label: 'PowerShell', code: 'Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body \'' + oneLine + '\'' },
        { id: 'curl', label: 'curl', code: 'curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d \'' + oneLine + '\'' },
        { id: 'docker', label: 'Docker', code: 'docker build -t my-model-api .\ndocker run -p 8000:8000 my-model-api' },
      ];
    });
    async function downloadExport() {
      await exportModel();
    }

    function onBatch(event) { inference.batchFile = event.target.files[0] || null; }
    function onImage(event) { setImage(event.target.files[0]); }

    return {
      inference, isTabular, isText, isImage, codeTabs, columnChoices, predict, predictBatch, explain, exportModel,
      exportOpen, apiTabs, downloadExport,
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
                <button v-if="isTabular" class="btn btn-secondary" @click="exportOpen = true"
                        title="Turn this model into a web service that other programs can call">
                  <Icon name="deployed_code" />Export as API
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

      <ModalDialog :open="exportOpen" title="Export as API" @close="exportOpen = false">
        <div class="stack">
          <p class="muted text-sm">Downloads <strong>{{ modelLabel(inference.modelName) }}</strong> as a small web service (a .zip).
            It runs on its own, without Easy-ML Studio, on any computer or server with Python or Docker.
            Other programs send it a row of data and get the prediction back.</p>
          <div class="stack-xs">
            <div class="field-label">In the zip</div>
            <ul class="hint-list text-sm muted">
              <li>the trained model and its preprocessing</li>
              <li><code>main.py</code>: the web service (FastAPI)</li>
              <li><code>requirements.txt</code> with the exact library versions, and a <code>Dockerfile</code></li>
              <li><code>README.md</code> with these steps and an example for this model</li>
            </ul>
          </div>
          <div class="stack-xs">
            <div class="alert alert-info"><Icon name="info" />
              <div>Only need it on this computer? <code>easy-ml serve {{ inference.runId }}</code> serves the same API
                straight from Easy-ML, without unzipping anything (see the "Serve as API" tab below the form).</div></div>
            <div class="field-label">How to use it</div>
            <CodeTabs :key="'api' + inference.runId + inference.modelName" :tabs="apiTabs" />
            <p class="subtle text-xs">The example uses {{ inference.columns.length }} input column{{ inference.columns.length === 1 ? '' : 's' }}
              from this run, with the values in the form, or a row of your training data. Missing columns count as missing values.</p>
          </div>
        </div>
        <template #footer>
          <button class="btn btn-secondary" @click="exportOpen = false">Close</button>
          <button class="btn btn-primary" @click="downloadExport" :disabled="inference.exporting">
            <Spinner v-if="inference.exporting" /><Icon v-else name="download" />Download .zip</button>
        </template>
      </ModalDialog>
    </div>`,
};
