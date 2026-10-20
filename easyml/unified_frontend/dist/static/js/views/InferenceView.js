import { computed, nextTick, ref } from '../lib/vue.js';
import { formatNumber, formatPercent, formatPrediction, formatScore } from '../lib/format.js';
import { featureAvailable, installHint, modelLabel, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import {
  allInputsEmpty, columnChoices, explain, exportModel, inference, inputKind, predict, predictBatch, sampleInForm, setImage,
  targetList, tryRealRow,
} from '../store/inference.js';

// Did the model get a sampled row's real answer right? (Regression: how far off it was.)
function verdictFor(predicted, actual, regression) {
  if (regression) return { regression: true, text: 'Off by ' + formatNumber(Math.abs(Number(predicted) - Number(actual)), 4) };
  const same = String(predicted) === String(actual)
    || (!Number.isNaN(Number(predicted)) && !Number.isNaN(Number(actual)) && Number(predicted) === Number(actual));
  return { correct: same, text: same ? 'Correct' : 'Wrong' };
}

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
    const targets = computed(() => targetList()); // the columns it predicts, when it predicts several
    const several = computed(() => targets.value.length > 0);

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
          'print(answer)   # ' + (several.value
            ? '{"prediction": {' + targets.value.map(t => '"' + t + '": ...').join(', ') + '}}'
            : (inference.task === 'MLRegression' ? '{"prediction": 123.4}' : '{"prediction": ..., "confidence": 0.93}')),
        ].join('\n') },
        { id: 'powershell', label: 'PowerShell', code: 'Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body \'' + oneLine + '\'' },
        { id: 'curl', label: 'curl', code: 'curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d \'' + oneLine + '\'' },
        { id: 'docker', label: 'Docker', code: 'docker build -t my-model-api .\ndocker run -p 8000:8000 my-model-api' },
      ];
    });
    async function downloadExport() {
      await exportModel();
    }

    function onBatch(event) {
      inference.batchFile = event.target.files[0] || null;
      inference.dataCheck = null; // it described the previous file
    }
    function onImage(event) { setImage(event.target.files[0]); }

    // Predicting a form with no values only shows what the model does with missing data.
    const allEmpty = computed(() => isTabular.value && inference.mode === 'manual' && allInputsEmpty());
    const canSample = computed(() => isText.value || (isTabular.value && inference.mode === 'manual'));

    // The real answer of a row taken from the data, while the form still holds that row.
    const sample = computed(() => (inference.result ? sampleInForm() : null));
    const verdict = computed(() => {
      const s = sample.value;
      if (!s || several.value) return null;
      return verdictFor(inference.result.prediction, s.actual, inference.task === 'MLRegression');
    });

    // Several columns to predict: one answer (and confidence, and real value for a sampled row) per column.
    const answers = computed(() => {
      const r = inference.result;
      if (!r || !several.value || !r.prediction || typeof r.prediction !== 'object') return [];
      const actual = sample.value ? sample.value.actual || {} : null;
      return targets.value.map(t => ({
        target: t,
        value: formatPrediction(r.prediction[t]),
        confidence: r.probability && r.probability[t] != null ? r.probability[t] : null,
        actual: actual ? formatPrediction(actual[t]) : null,
        verdict: actual && actual[t] != null ? verdictFor(r.prediction[t], actual[t], inference.task === 'MLRegression') : null,
      }));
    });
    function onExplainTarget() {
      inference.explanation = null; // it explained another column
      inference.shapPlot = null;
    }

    const explainCard = ref(null);
    const plotOpen = ref(false);
    async function explainAndShow() {
      await explain();
      await nextTick();
      if (inference.explanation && explainCard.value) explainCard.value.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    const predictedLabel = computed(() => {
      const r = inference.result;
      if (!r) return '';
      return formatPrediction(several.value && r.prediction ? r.prediction[inference.explainTarget] : r.prediction);
    });
    const explanationSummary = computed(() => {
      const e = inference.explanation;
      if (!e) return '';
      const label = '“' + predictedLabel.value + '”';
      if (e.scale === 'probability') {
        return 'On average the model gives ' + label + ' ' + formatPercent(e.base_value) + '. With these inputs it gives '
          + formatPercent(e.output_value) + '.';
      }
      if (e.scale === 'log-odds') {
        return 'The model scores ' + label + ' in log-odds (higher means more likely): ' + formatNumber(e.base_value, 3)
          + ' on average, ' + formatNumber(e.output_value, 3) + ' with these inputs.';
      }
      return 'The average prediction is ' + formatNumber(e.base_value, 4) + '. These inputs move it to '
        + formatNumber(e.output_value, 4) + '.';
    });
    const axisTitle = computed(() => {
      const e = inference.explanation;
      if (!e) return '';
      const label = '“' + predictedLabel.value + '”';
      if (e.scale === 'probability') return 'Change in the probability of ' + label + ' (percentage points)';
      if (e.scale === 'log-odds') return 'Change in the score for ' + label + ' (log-odds)';
      return 'Change in the predicted value';
    });
    const explainHint = computed(() => (inference.task === 'MLRegression'
      ? 'Red bars raise the prediction, blue bars lower it. Strongest effects first.'
      : 'Red bars push towards “' + predictedLabel.value + '”, blue bars push away from it. Strongest effects first.'));
    const hasMissing = computed(() => !!inference.explanation && inference.explanation.features.some(f => f.missing));

    // The data-prep recipe of a run trained on cleaned data, in one sentence.
    const cleaningSteps = computed(() => {
      const c = inference.cleaning;
      if (!c || !c.steps || !c.steps.length) return '';
      const shown = c.steps.slice(0, 6).join(', ');
      return c.steps.length > 6 ? shown + ' and ' + (c.steps.length - 6) + ' more' : shown;
    });
    const resultWarnings = computed(() => (inference.result && inference.result.warnings) || []);
    const flaggedColumns = computed(() => (inference.dataCheck ? inference.dataCheck.columns.filter(c => c.status !== 'ok') : []));

    return {
      inference, isTabular, isText, isImage, codeTabs, columnChoices, predict, predictBatch, explain, exportModel,
      exportOpen, apiTabs, downloadExport, allEmpty, canSample, tryRealRow, sample, verdict,
      explainCard, plotOpen, explainAndShow, predictedLabel, explanationSummary, axisTitle, explainHint, hasMissing,
      cleaningSteps, resultWarnings, flaggedColumns, inputKind, targets, several, answers, onExplainTarget,
      onBatch, onImage, featureAvailable, installHint, modelLabel, taskLabel, navigate, formatPrediction, formatPercent, formatScore,
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
                <span v-if="several && inference.modelName === 'best_for_each_target' && inference.targets.combined" class="field-hint">
                  <template v-for="(t, i) in targets" :key="t">{{ i ? ', ' : '' }}{{ t }}: {{ modelLabel(inference.targets.combined[t]) }}</template>
                </span>
              </div>

              <div v-if="inference.cleaning && inference.cleaning.automatic && !isImage" class="alert alert-info">
                <Icon name="cleaning_services" />
                <div>Trained on cleaned data: your input is cleaned with the recipe <strong>{{ inference.cleaning.name }}</strong>
                  ({{ cleaningSteps }}) before predicting, using the values learned from the training data.
                  Enter the data as it originally looks.</div>
              </div>
              <div v-else-if="inference.cleaning && !isImage" class="alert alert-warning">
                <Icon name="warning_amber" />
                <div>This run was trained on data cleaned with the recipe <strong>{{ inference.cleaning.name }}</strong>
                  before Easy-ML 0.11, so it does not clean input itself. Enter values as they look after cleaning,
                  or retrain the run to have input cleaned automatically.</div>
              </div>

              <template v-if="isTabular">
                <div class="segmented" role="group" aria-label="Input mode">
                  <button :class="{ active: inference.mode === 'manual' }" @click="inference.mode = 'manual'"><Icon name="edit_note" size="sm" />Single row</button>
                  <button :class="{ active: inference.mode === 'batch' }" @click="inference.mode = 'batch'"><Icon name="upload_file" size="sm" />CSV file</button>
                </div>
                <div v-if="inference.mode === 'manual'" class="form-grid">
                  <div v-for="col in inference.columns" :key="col" class="field">
                    <label :for="'in-' + col" class="truncate" :title="col">{{ col }}</label>
                    <select v-if="inputKind(col) === 'choice'" :id="'in-' + col" class="select" v-model="inference.values[col]">
                      <option value="">(missing)</option>
                      <option v-for="v in columnChoices(col)" :key="String(v)" :value="v">{{ v }}</option>
                    </select>
                    <input v-else-if="inputKind(col) === 'text'" :id="'in-' + col" class="input" type="text" v-model="inference.values[col]" placeholder="text" />
                    <input v-else :id="'in-' + col" class="input" type="number" step="any" v-model="inference.values[col]" placeholder="number" />
                  </div>
                </div>
                <div v-else class="field">
                  <label for="batch-file">{{ inference.cleaning && inference.cleaning.automatic ? 'CSV with the original columns' : 'CSV with the same columns' }}</label>
                  <input id="batch-file" class="input" type="file" accept=".csv" @change="onBatch" />
                  <span class="field-hint">{{ several ? 'You get the file back with a Prediction column for each column to predict.'
                    : 'You get the file back with a Prediction column added.' }}</span>
                </div>
                <p class="field-hint" v-if="allEmpty">Fill in at least one value, or press <strong>Try a real row</strong>
                  to fill the form with a row from this run's data.</p>
                <p class="field-hint" v-else-if="inference.mode === 'manual'">Leave a field empty and it is filled in the same way missing values were during training.</p>
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
                <button v-if="!isTabular || inference.mode === 'manual'" class="btn btn-primary" @click="predict"
                        :disabled="inference.predicting || allEmpty" :title="allEmpty ? 'Fill in at least one value first' : ''">
                  <Spinner v-if="inference.predicting" /><Icon v-else name="bolt" />Predict
                </button>
                <button v-else class="btn btn-primary" @click="predictBatch" :disabled="inference.predicting || !inference.batchFile">
                  <Spinner v-if="inference.predicting" /><Icon v-else name="download" />Predict and download
                </button>
                <button v-if="canSample" class="btn btn-secondary" @click="tryRealRow" :disabled="inference.sampling || inference.predicting"
                        title="Fill the form with a random row of this run's data (from its test set when available) and predict it">
                  <Spinner v-if="inference.sampling" /><Icon v-else name="shuffle" />Try a real row
                </button>
                <button v-if="isTabular" class="btn btn-secondary" @click="exportOpen = true"
                        title="Turn this model into a web service that other programs can call">
                  <Icon name="deployed_code" />Export as API
                </button>
              </div>
            </div>
          </section>
        </div>

        <aside class="card inference-result">
          <div class="card-header"><h3>Prediction</h3></div>
          <div class="card-body stack">
            <template v-if="inference.result && several">
              <div class="target-answers">
                <div v-for="a in answers" :key="a.target" class="target-answer">
                  <span class="target-name">{{ a.target }}</span>
                  <span class="target-value">{{ a.value }}</span>
                  <span v-if="a.confidence != null" class="subtle text-xs">{{ formatPercent(a.confidence) }} confident</span>
                  <span v-if="a.verdict" class="row-nowrap text-xs"><span class="muted">In the data: {{ a.actual }}</span>
                    <span class="badge" :class="a.verdict.regression ? '' : (a.verdict.correct ? 'badge-success' : 'badge-danger')">{{ a.verdict.text }}</span></span>
                </div>
              </div>
              <p v-if="sample" class="subtle text-xs">Row {{ (sample.row + 1).toLocaleString() }} of the run's data,
                {{ sample.source === 'test' ? 'from its test set: the model never trained on it.' : 'used for training.' }}</p>
              <div v-if="resultWarnings.length" class="alert alert-warning text-sm">
                <Icon name="warning_amber" />
                <div class="stack-xs"><strong>Unlike the training data</strong>
                  <span v-for="(w, i) in resultWarnings" :key="i">{{ w }}</span>
                  <span class="subtle text-xs">The prediction may be less reliable.</span></div>
              </div>
              <div class="field">
                <label for="explain-target">Explain the prediction of</label>
                <select id="explain-target" class="select" v-model="inference.explainTarget" @change="onExplainTarget">
                  <option v-for="t in targets" :key="t" :value="t">{{ t }}</option>
                </select>
              </div>
              <button class="btn btn-secondary" @click="explainAndShow" :disabled="inference.explaining || !featureAvailable('explain')">
                <Spinner v-if="inference.explaining" /><Icon v-else name="insights" />Explain this prediction
              </button>
              <p v-if="!featureAvailable('explain')" class="field-hint">Explanations need SHAP: run <code>{{ installHint('explain') }}</code>,
                then restart the studio.</p>
            </template>
            <template v-else-if="inference.result">
              <div class="prediction-value">{{ formatPrediction(inference.result.prediction) }}</div>
              <div v-if="inference.result.probability != null" class="stack-sm">
                <div class="row-between text-sm"><span class="muted">Confidence</span><span class="strong num">{{ formatPercent(inference.result.probability) }}</span></div>
                <div class="progress"><div class="progress-bar" :style="{ width: (inference.result.probability * 100) + '%' }"></div></div>
              </div>
              <div v-if="sample" class="sample-check stack-xs">
                <div class="row-between text-sm"><span class="muted">In the data</span>
                  <span class="row-nowrap"><span class="strong">{{ formatPrediction(sample.actual) }}</span>
                    <span class="badge" :class="verdict.regression ? '' : (verdict.correct ? 'badge-success' : 'badge-danger')">
                      <Icon :name="verdict.regression ? 'straighten' : (verdict.correct ? 'check_circle' : 'cancel')" />{{ verdict.text }}</span></span></div>
                <p class="subtle text-xs">Row {{ (sample.row + 1).toLocaleString() }} of the run's data,
                  {{ sample.source === 'test' ? 'from its test set: the model never trained on it.' : 'used for training.' }}</p>
              </div>
              <div v-if="resultWarnings.length" class="alert alert-warning text-sm">
                <Icon name="warning_amber" />
                <div class="stack-xs"><strong>Unlike the training data</strong>
                  <span v-for="(w, i) in resultWarnings" :key="i">{{ w }}</span>
                  <span class="subtle text-xs">The prediction may be less reliable.</span></div>
              </div>
              <template v-if="isTabular">
                <button class="btn btn-secondary" @click="explainAndShow" :disabled="inference.explaining || !featureAvailable('explain')">
                  <Spinner v-if="inference.explaining" /><Icon v-else name="insights" />Explain this prediction
                </button>
                <p v-if="!featureAvailable('explain')" class="field-hint">Explanations need SHAP: run <code>{{ installHint('explain') }}</code>,
                  then restart the studio.</p>
              </template>
            </template>
            <p v-else class="muted text-sm">{{ isImage ? 'Choose an image and press Predict.' : 'Fill in the inputs and press Predict, or try a real row from the data.' }}</p>
          </div>
        </aside>
      </div>

      <section v-if="isTabular && inference.mode === 'batch' && (inference.checking || inference.dataCheck)" class="card">
        <div class="card-header">
          <div><h3>Data check</h3>
            <div class="card-hint">Does the file you predicted look like the data the model was trained on?</div></div>
          <span v-if="inference.dataCheck" class="badge"
                :class="inference.dataCheck.status === 'ok' ? 'badge-success' : (inference.dataCheck.status === 'major' ? 'badge-danger' : 'badge-warning')">
            {{ inference.dataCheck.status === 'ok' ? 'Looks alike' : (inference.dataCheck.status === 'major' ? 'Looks different' : 'Slightly different') }}</span>
        </div>
        <div class="card-body stack-sm">
          <p v-if="inference.checking" class="muted text-sm"><Spinner /> Comparing with the training data...</p>
          <template v-else>
            <p class="text-sm">{{ inference.dataCheck.summary }}</p>
            <div v-if="flaggedColumns.length" class="table-wrap">
              <table class="table">
                <thead><tr><th>Column</th><th>Status</th><th>What changed</th></tr></thead>
                <tbody>
                  <tr v-for="c in flaggedColumns" :key="c.column">
                    <td class="strong">{{ c.column }}</td>
                    <td><span class="badge" :class="c.status === 'major' ? 'badge-danger' : 'badge-warning'">{{ c.status === 'major' ? 'Different' : 'Slightly' }}</span></td>
                    <td class="text-sm">{{ c.message }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p v-if="inference.dataCheck.missing_columns.length" class="subtle text-xs">Not in the file:
              {{ inference.dataCheck.missing_columns.join(', ') }}</p>
            <p v-if="inference.dataCheck.status !== 'ok'" class="subtle text-xs">Predictions for data unlike the training
              data are less reliable. If this is the new normal, retrain the run on recent data.</p>
          </template>
        </div>
      </section>

      <section v-if="inference.explanation && inference.result" ref="explainCard" class="card explain-card">
        <div class="card-header">
          <div><h3>Why the model predicted {{ predictedLabel }}<template v-if="several"> for {{ inference.explainTarget }}</template></h3>
            <div class="card-hint">{{ explanationSummary }}</div></div>
          <button v-if="inference.shapPlot" class="btn btn-ghost btn-sm" @click="plotOpen = true"><Icon name="zoom_in" />Detailed SHAP plot</button>
        </div>
        <div class="card-body stack-sm">
          <ContributionChart :data="inference.explanation" :axis-title="axisTitle" />
          <p class="field-hint">{{ explainHint }}<span v-if="hasMissing"> Columns shown as missing were filled in the same way as during training.</span></p>
        </div>
      </section>

      <section class="card">
        <div class="card-header"><div><h3>Use this model in code</h3>
          <div class="card-hint">Run on this computer, with the same Easy-ML data directory (set EASYML_DATA_DIR if you changed it).</div></div></div>
        <div class="card-body"><CodeTabs :key="inference.runId + inference.modelName" :tabs="codeTabs" /></div>
      </section>

      <ModalDialog :open="plotOpen" xl title="SHAP waterfall plot" @close="plotOpen = false">
        <div class="stack-sm">
          <p class="muted text-sm">SHAP's own view of the same prediction, by the columns the model sees after preprocessing
            (scaled numbers, and one column per category).</p>
          <div class="figure figure-full"><img :src="'data:image/png;base64,' + inference.shapPlot" alt="SHAP waterfall plot" /></div>
        </div>
        <template #footer>
          <a class="btn btn-secondary" :href="'data:image/png;base64,' + inference.shapPlot" :download="'shap_run_' + inference.runId + '.png'">
            <Icon name="download" />Download PNG</a>
          <button class="btn btn-primary" @click="plotOpen = false">Close</button>
        </template>
      </ModalDialog>

      <ModalDialog :open="exportOpen" title="Export as API" @close="exportOpen = false">
        <div class="stack">
          <p v-if="several" class="muted text-sm">Downloads <strong>{{ modelLabel(inference.modelName) }}</strong> as a small web
            service (a .zip) that predicts {{ targets.join(', ') }} with one model per column. It runs without the studio
            on any computer or server with Python or Docker; its requirements install the easy-ml-studio package for the
            code that holds the models together. Other programs send it a row of data and get the predictions back.</p>
          <p v-else class="muted text-sm">Downloads <strong>{{ modelLabel(inference.modelName) }}</strong> as a small web service (a .zip).
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
