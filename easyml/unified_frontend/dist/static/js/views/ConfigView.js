import { featureAvailable, installHint, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import { featureColumns, isClassification, isImage, isTabular, isText, severalTargets, wizard } from '../store/wizard.js';

const IMBALANCE = [
  { id: 'none', label: 'None' },
  { id: 'smote', label: 'SMOTE', hint: 'oversample' },
  { id: 'tomek', label: 'Tomek links', hint: 'undersample' },
  { id: 'smotetomek', label: 'SMOTE + Tomek', hint: 'hybrid' },
];

export default {
  setup() {
    function toggle(list, column) {
      const values = wizard.config[list];
      const idx = values.indexOf(column);
      if (idx >= 0) values.splice(idx, 1); else values.push(column);
    }
    return {
      wizard, featureColumns, isClassification, isImage, isTabular, isText, IMBALANCE, toggle, severalTargets,
      featureAvailable, installHint, taskLabel, navigate,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader :eyebrow="'Step 2 · ' + taskLabel(wizard.config.task)" title="Configure the pipeline"
                  subtitle="Sensible defaults are already set. Adjust them only if you need to." />

      <!-- Tabular -->
      <template v-if="isTabular">
        <section class="card">
          <div class="card-header"><h3>Data preparation</h3></div>
          <div class="card-body form-grid">
            <div class="field">
              <label>Missing values</label>
              <select class="select" v-model="wizard.config.imputation_strategy">
                <option value="impute">Fill in automatically (median / most frequent)</option>
                <option value="drop">Drop rows with missing values</option>
              </select>
            </div>
            <div class="field">
              <label>Scale numeric features</label>
              <select class="select" v-model="wizard.config.scaling">
                <option value="No">No</option>
                <option value="Yes">Yes (standardize)</option>
              </select>
              <span class="field-hint">Helps linear models and KNN.</span>
            </div>
            <div class="field">
              <label>Reduce features</label>
              <select class="select" v-model="wizard.config.reduce_features">
                <option value="No">No</option>
                <option value="Yes">Yes (PCA, keeps 95% of variance)</option>
              </select>
            </div>
          </div>
        </section>

        <section class="card">
          <div class="card-header"><div><h3>Columns</h3><div class="card-hint">Categorical columns are detected automatically. Override the detection or leave columns out.</div></div></div>
          <div class="card-body stack">
            <div class="stack-sm">
              <div class="field-label">Treat as categorical <span class="subtle text-xs">(optional)</span></div>
              <div class="chip-list">
                <button v-for="col in featureColumns" :key="col" class="chip" :class="{ selected: wizard.config.categorical_columns.includes(col) }"
                        @click="toggle('categorical_columns', col)" :aria-pressed="wizard.config.categorical_columns.includes(col)">{{ col }}</button>
              </div>
            </div>
            <div class="stack-sm">
              <div class="field-label">Ignore columns <span class="subtle text-xs">(e.g. IDs or names)</span></div>
              <div class="chip-list">
                <button v-for="col in featureColumns" :key="col" class="chip" :class="{ selected: wizard.config.ignore_columns.includes(col) }"
                        @click="toggle('ignore_columns', col)" :aria-pressed="wizard.config.ignore_columns.includes(col)">{{ col }}</button>
              </div>
            </div>
          </div>
        </section>
      </template>

      <!-- Text -->
      <template v-if="isText">
        <section class="card">
          <div class="card-header"><h3>Text cleaning</h3></div>
          <div class="card-body grid-auto">
            <label class="check" :class="{ checked: wizard.config.lowercase }">
              <input type="checkbox" v-model="wizard.config.lowercase" />
              <div><div class="check-title">Lowercase</div><div class="check-hint">"Great" and "great" become the same word.</div></div>
            </label>
            <label class="check" :class="{ checked: wizard.config.remove_stopwords }">
              <input type="checkbox" v-model="wizard.config.remove_stopwords" />
              <div><div class="check-title">Remove stopwords</div><div class="check-hint">Drop common words like "the" and "is".</div></div>
            </label>
            <label class="check" :class="{ checked: wizard.config.regex_clean }">
              <input type="checkbox" v-model="wizard.config.regex_clean" />
              <div><div class="check-title">Remove symbols</div><div class="check-hint">Keep only letters, digits and spaces.</div></div>
            </label>
            <label class="check" :class="{ checked: wizard.config.lemmatize, disabled: !featureAvailable('lemmatize') }"
                   :title="featureAvailable('lemmatize') ? '' : 'Needs: ' + installHint('lemmatize')">
              <input type="checkbox" v-model="wizard.config.lemmatize" :disabled="!featureAvailable('lemmatize')" />
              <div><div class="check-title">Lemmatize</div>
                <div class="check-hint" v-if="featureAvailable('lemmatize')">"running" becomes "run".</div>
                <div class="check-hint" v-else>Needs <code>{{ installHint('lemmatize') }}</code></div></div>
            </label>
          </div>
        </section>

        <section class="card">
          <div class="card-header"><h3>Text features</h3></div>
          <div class="card-body stack">
            <div class="segmented" role="group" aria-label="Feature extraction">
              <button :class="{ active: wizard.config.feature_mode === 'vectorizer' }" @click="wizard.config.feature_mode = 'vectorizer'">Word counts (TF-IDF)</button>
              <button :class="{ active: wizard.config.feature_mode === 'embedding' }" :disabled="!featureAvailable('embeddings')"
                      @click="wizard.config.feature_mode = 'embedding'">Sentence embeddings</button>
            </div>
            <p v-if="!featureAvailable('embeddings')" class="field-hint">Sentence embeddings need <code>{{ installHint('embeddings') }}</code></p>
            <div v-if="wizard.config.feature_mode === 'vectorizer'" class="form-grid">
              <div class="field">
                <label>Method</label>
                <select class="select" v-model="wizard.config.vectorizer_type">
                  <option value="tfidf">TF-IDF</option>
                  <option value="count">Word counts</option>
                  <option value="hashing">Hashing</option>
                </select>
              </div>
              <div class="field">
                <label>Vocabulary size</label>
                <input class="input" type="number" min="100" step="500" v-model.number="wizard.config.max_features" />
              </div>
            </div>
            <div v-else class="form-grid">
              <div class="field">
                <label>Embedding model</label>
                <select class="select" v-model="wizard.config.embedding_model">
                  <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (fast)</option>
                  <option value="all-mpnet-base-v2">all-mpnet-base-v2 (more accurate)</option>
                </select>
              </div>
            </div>
          </div>
        </section>
      </template>

      <!-- Images -->
      <section v-if="isImage" class="card">
        <div class="card-header"><h3>Image preprocessing</h3></div>
        <div class="card-body form-grid">
          <div class="field">
            <label>Image size</label>
            <select class="select" v-model="wizard.config.image_size_input">
              <option value="224,224">224 × 224 (best for ResNet / MobileNet)</option>
              <option value="128,128">128 × 128 (faster)</option>
              <option value="64,64">64 × 64 (fastest)</option>
            </select>
          </div>
          <div class="field">
            <label>Batch size</label>
            <input class="input" type="number" min="1" v-model.number="wizard.config.batch_size" />
          </div>
          <div class="field">
            <label>Validation share</label>
            <input class="input" type="number" min="0.05" max="0.5" step="0.05" v-model.number="wizard.config.validation_split" />
          </div>
          <div class="field">
            <label>Augmentation</label>
            <select class="select" v-model="wizard.config.augment">
              <option :value="true">On (random flips and rotations)</option>
              <option :value="false">Off</option>
            </select>
          </div>
        </div>
      </section>

      <!-- Class balance (classification only) -->
      <section v-if="isClassification && severalTargets" class="card">
        <div class="card-header"><h3>Class balance</h3></div>
        <div class="card-body">
          <p class="muted">Rebalancing (SMOTE, Tomek links) is not available when several columns are predicted: it
            changes rows to balance one column, which would unbalance the others.</p>
        </div>
      </section>
      <section v-else-if="isClassification" class="card">
        <div class="card-header"><h3>Class balance</h3></div>
        <div class="card-body grid-2">
          <ClassDistribution :data="wizard.classDistribution" :loading="wizard.classDistributionLoading" />
          <div class="field">
            <label>Rebalance training data</label>
            <div class="segmented segmented-block" role="group" aria-label="Rebalancing">
              <button v-for="opt in IMBALANCE" :key="opt.id" :class="{ active: wizard.config.imbalance_handling === opt.id }"
                      @click="wizard.config.imbalance_handling = opt.id">{{ opt.label }}<small v-if="opt.hint">{{ opt.hint }}</small></button>
            </div>
            <span class="field-hint">Applied to the training folds only, so test scores stay honest.</span>
          </div>
        </div>
      </section>

      <div class="row-between">
        <button class="btn btn-ghost" @click="navigate('upload')"><Icon name="arrow_back" />Back</button>
        <button class="btn btn-primary" @click="navigate('models')">Choose models<Icon name="arrow_forward" /></button>
      </div>
    </div>`,
};
