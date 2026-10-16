import { computed, onMounted } from '../lib/vue.js';
import { featureAvailable, installHint, modelLabel, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import { availableModels, initialiseModelSelection, isImage, isTabular, isText, wizard } from '../store/wizard.js';
import { startTraining } from '../store/training.js';

const MODEL_HINTS = {
  logistic_regression: 'Fast linear baseline', linear_regression: 'Fast linear baseline',
  random_forest: 'Robust all-rounder', xg_boost: 'Gradient boosting, often top accuracy',
  light_gbm: 'Fast gradient boosting', cat_boost: 'Boosting with strong categorical handling',
  knn: 'Similarity-based', ada_boost: 'Boosted decision stumps',
  naive_bayes: 'Classic for word counts', complement_nb: 'Naive Bayes for imbalanced text',
  svm: 'Strong linear text classifier', mlp: 'Small neural network',
  SimpleCNN: 'Small network trained from scratch', ResNet18: 'Pretrained, accurate (downloads weights once)',
  MobileNetV2: 'Pretrained, light and fast (downloads weights once)', CustomCNN: 'Design your own layers',
};

const CV_FOLDS = [
  { id: 0, label: 'Off', hint: 'Test set only, fastest' },
  { id: 3, label: '3 folds', hint: 'Each model trains 3 more times' },
  { id: 5, label: '5 folds', hint: 'The usual choice' },
  { id: 10, label: '10 folds', hint: 'Most reliable, slowest' },
];

const TUNING = [
  { id: 0, label: 'None', hint: 'Default settings, fastest' },
  { id: 1, label: 'Light', hint: 'Small grid search' },
  { id: 2, label: 'Thorough', hint: 'Large grid, much slower' },
];

export default {
  setup() {
    onMounted(initialiseModelSelection);
    const models = computed(availableModels);
    const allSelected = computed(() => models.value.length > 0 && models.value.every(m => wizard.config.models.includes(m)));
    function toggleAll() {
      wizard.config.models = allSelected.value ? [] : [...models.value];
    }
    const missingExtra = computed(() => (isImage.value && !featureAvailable('cv')) || !models.value.length);
    return {
      wizard, models, allSelected, toggleAll, missingExtra, MODEL_HINTS, TUNING, CV_FOLDS, isImage, isTabular, isText,
      featureAvailable, installHint, modelLabel, taskLabel, navigate, startTraining,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader :eyebrow="'Step 3 · ' + taskLabel(wizard.config.task)" title="Choose models"
                  subtitle="Every selected model is trained on the same data and ranked on a held-out test set." />

      <div v-if="missingExtra" class="alert alert-warning">
        <Icon name="extension" />
        <div>This task needs an optional extra. Stop the studio, run <code>{{ installHint(isImage ? 'cv' : 'embeddings') }}</code>, then start it again.</div>
      </div>

      <section v-else class="card">
        <div class="card-header">
          <div><h3>Models</h3><div class="card-hint">{{ wizard.config.models.length }} of {{ models.length }} selected</div></div>
          <button class="btn btn-ghost btn-sm" @click="toggleAll">{{ allSelected ? 'Clear all' : 'Select all' }}</button>
        </div>
        <div class="card-body stack">
          <div class="model-grid">
            <label v-for="m in models" :key="m" class="check" :class="{ checked: wizard.config.models.includes(m) }">
              <input type="checkbox" :value="m" v-model="wizard.config.models" />
              <div><div class="check-title">{{ modelLabel(m) }}</div><div class="check-hint">{{ MODEL_HINTS[m] || '' }}</div></div>
            </label>
          </div>
          <p v-if="isTabular && !featureAvailable('boost')" class="field-hint">
            XGBoost, LightGBM and CatBoost are available with <code>{{ installHint('boost') }}</code>
          </p>
        </div>
      </section>

      <section v-if="!isImage" class="card">
        <div class="card-header"><div><h3>Hyperparameter tuning</h3><div class="card-hint">Grid search with 4-fold cross-validation on the training data.</div></div></div>
        <div class="card-body">
          <div class="segmented segmented-block" role="group" aria-label="Tuning level">
            <button v-for="t in TUNING" :key="t.id" :class="{ active: wizard.config.hyper_parameter_tuning === t.id }"
                    @click="wizard.config.hyper_parameter_tuning = t.id">{{ t.label }}<small>{{ t.hint }}</small></button>
          </div>
        </div>
      </section>

      <section v-if="isImage && !missingExtra" class="card">
        <div class="card-header"><h3>Training</h3></div>
        <div class="card-body form-grid">
          <div class="field"><label>Epochs</label><input class="input" type="number" min="1" v-model.number="wizard.config.epochs" /></div>
          <div class="field"><label>Learning rate</label><input class="input" type="number" step="0.0001" min="0" v-model.number="wizard.config.learning_rate" /></div>
          <div class="field">
            <label>Early stopping patience</label>
            <input class="input" type="number" min="1" v-model.number="wizard.config.early_stopping_patience" />
            <span class="field-hint">Stop after this many epochs without improvement.</span>
          </div>
          <div class="field">
            <label>Reduce learning rate after</label>
            <input class="input" type="number" min="1" v-model.number="wizard.config.reduce_lr_patience" />
            <span class="field-hint">epochs without improvement</span>
          </div>
          <div class="field">
            <label>Reduction factor</label>
            <input class="input" type="number" step="0.1" min="0.1" max="0.9" v-model.number="wizard.config.reduce_lr_factor" />
          </div>
        </div>
      </section>

      <CnnBuilder v-if="isImage && wizard.config.models.includes('CustomCNN')" />

      <section v-if="!missingExtra" class="card">
        <div class="card-header"><div><h3>Validation and reproducibility</h3>
          <div class="card-hint">The test set ranks the models. Cross-validation also shows how much a score changes
            on different parts of the training data.</div></div></div>
        <div class="card-body stack">
          <div v-if="!isImage" class="field">
            <span class="field-label">Cross-validation</span>
            <div class="segmented segmented-block" role="group" aria-label="Cross-validation folds">
              <button v-for="f in CV_FOLDS" :key="f.id" :class="{ active: wizard.config.cv_folds === f.id }"
                      @click="wizard.config.cv_folds = f.id">{{ f.label }}<small>{{ f.hint }}</small></button>
            </div>
          </div>
          <div class="field">
            <label for="random-seed">Random seed</label>
            <input id="random-seed" class="input input-number" type="number" min="0" step="1" v-model.number="wizard.config.random_seed" />
            <span class="field-hint">The same data, settings and seed always give the same models and scores.
              Try another seed to see how much the results depend on chance.</span>
          </div>
        </div>
      </section>

      <div class="row-between">
        <button class="btn btn-ghost" @click="navigate('config')"><Icon name="arrow_back" />Back</button>
        <button class="btn btn-primary btn-lg" :disabled="!wizard.config.models.length || missingExtra" @click="startTraining">
          <Icon name="play_arrow" />Start training
        </button>
      </div>
    </div>`,
};
