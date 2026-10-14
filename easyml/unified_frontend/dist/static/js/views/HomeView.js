import { computed, onMounted } from '../lib/vue.js';
import { formatScore, relativeTime } from '../lib/format.js';
import { featureAvailable, installHint, session } from '../store/session.js';
import { navigate, toastError, confirmAction } from '../store/ui.js';
import { resetWizard, wizard } from '../store/wizard.js';
import { isRunning, resetTraining, training } from '../store/training.js';
import { loadRuns, openRunDetail, runs } from '../store/runs.js';
import { openInference } from '../store/inference.js';

const TASKS = [
  { id: 'MLClassification', icon: 'category', title: 'Tabular classification', text: 'Predict a category from a CSV or Excel table, such as churn, fraud or diagnosis.', tags: ['Random Forest', 'XGBoost', 'Logistic Regression'] },
  { id: 'MLRegression', icon: 'trending_up', title: 'Tabular regression', text: 'Predict a number from a table, such as a price, demand or a duration.', tags: ['Linear Regression', 'LightGBM', 'CatBoost'] },
  { id: 'NLPTextClassification', icon: 'article', title: 'Text classification', text: 'Sort text into labels, such as sentiment, topics or support tickets.', tags: ['TF-IDF', 'Embeddings'] },
  { id: 'ImageClassification', icon: 'image', title: 'Image classification', text: 'Train a CNN on folders of images, one folder per class.', tags: ['ResNet-18', 'MobileNet V2', 'Custom CNN'], feature: 'cv' },
];

const UPCOMING = [
  { icon: 'cleaning_services', title: 'Data prep (ETL)', text: 'Clean, transform and export a dataset without training a model.', badge: 'Planned' },
  { icon: 'timeline', title: 'Time-series forecasting', text: 'Forecast future values from dated data.', badge: 'Planned' },
  { icon: 'autorenew', title: 'Auto-retraining', text: 'Retrain a saved run on new data, on a schedule or a trigger.', badge: 'Planned' },
  { icon: 'psychology', title: 'LLM fine-tuning', text: 'Fine-tune small language models on your own text.', badge: 'Exploring' },
];

export default {
  setup() {
    onMounted(() => { if (!runs.loaded) loadRuns(); });
    const recent = computed(() => runs.list.slice(0, 4));

    async function start(task) {
      if (task.feature && !featureAvailable(task.feature)) {
        toastError(task.title + ' needs an optional extra: ' + installHint(task.feature));
        return;
      }
      if (isRunning()) {
        const ok = await confirmAction({
          title: 'Training in progress',
          message: 'A model is still training. Starting a new run stops following it here (it keeps running on the server).',
          confirmLabel: 'Start new run',
        });
        if (!ok) return;
      }
      if (training.status !== 'running') resetTraining();
      resetWizard(task.id);
      navigate('upload');
    }

    function openRun(run) {
      navigate('history');
      openRunDetail(run.id);
    }

    function scrollToTasks() {
      const el = document.getElementById('tasks');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }

    return {
      TASKS, UPCOMING, session, recent, runs, wizard, start, openRun, scrollToTasks, featureAvailable, installHint,
      navigate, formatScore, relativeTime, openInference,
    };
  },
  template: /* html */ `
    <div class="page">
      <section class="hero">
        <div>
          <div class="eyebrow">Open-source · runs on your machine</div>
          <h1>Train machine learning models without writing code</h1>
          <p>Upload a dataset, compare several models side by side, explain their predictions and export the best one as an API.</p>
          <div class="hero-actions">
            <button class="btn btn-primary btn-lg" @click="scrollToTasks"><Icon name="add" />New run</button>
            <button class="btn btn-secondary btn-lg" @click="navigate('history')"><Icon name="history" />Run history</button>
          </div>
        </div>
        <div class="hero-aside">
          <div class="field-label">Install anywhere with pip</div>
          <CopyField text="pip install easy-ml-studio" label="Copy install command" />
          <div class="hero-points">
            <div><Icon name="check_circle" />Your data never leaves your computer</div>
            <div><Icon name="check_circle" />Tabular, text and image models</div>
            <div><Icon name="check_circle" />Use trained models from Python or the command line</div>
          </div>
        </div>
      </section>

      <section id="tasks" class="stack">
        <div class="section-title">
          <h2>Start a new run</h2>
          <p>Pick what you want to predict.</p>
        </div>
        <div class="grid-4">
          <button v-for="task in TASKS" :key="task.id" class="task-card"
                  :class="{ unavailable: task.feature && !featureAvailable(task.feature) }"
                  :aria-disabled="task.feature && !featureAvailable(task.feature) ? 'true' : null"
                  @click="start(task)">
            <span class="task-icon"><Icon :name="task.icon" /></span>
            <h3>{{ task.title }}</h3>
            <p>{{ task.text }}</p>
            <div class="task-meta">
              <template v-if="task.feature && !featureAvailable(task.feature)">
                <span class="badge badge-warning"><Icon name="extension" />Needs extra</span>
                <code class="subtle text-xs">{{ installHint(task.feature) }}</code>
              </template>
              <span v-else v-for="tag in task.tags" :key="tag" class="badge">{{ tag }}</span>
            </div>
          </button>
        </div>
      </section>

      <section class="stack">
        <div class="section-title">
          <h2>Coming soon</h2>
          <p>On the roadmap for upcoming releases.</p>
        </div>
        <div class="grid-4">
          <div v-for="item in UPCOMING" :key="item.title" class="task-card upcoming" aria-disabled="true">
            <span class="task-icon"><Icon :name="item.icon" /></span>
            <h3>{{ item.title }}</h3>
            <p>{{ item.text }}</p>
            <div class="task-meta"><span class="badge" :class="item.badge === 'Planned' ? 'badge-accent' : ''">{{ item.badge }}</span></div>
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-header">
          <div>
            <h3>Recent runs</h3>
            <div class="card-hint">Your latest trained models.</div>
          </div>
          <button class="btn btn-ghost btn-sm" @click="navigate('history')">View all<Icon name="arrow_forward" /></button>
        </div>
        <div v-if="!recent.length" class="empty">
          <Icon name="model_training" />
          <h3>No runs yet</h3>
          <p>Train your first model to see it here.</p>
        </div>
        <div v-else class="recent-list">
          <button v-for="run in recent" :key="run.id" class="recent-item" @click="openRun(run)">
            <div class="truncate">
              <div class="run-name truncate">{{ run.run_name }}</div>
              <div class="run-sub"><TaskBadge :task="run.task" /><span class="truncate">{{ run.dataset_name }}</span></div>
            </div>
            <div class="right">
              <div class="best-score">{{ formatScore(run.best_score, run.metric) }}</div>
              <div class="subtle text-xs">{{ run.best_model_label || '—' }}</div>
            </div>
            <div class="subtle text-xs nowrap">{{ relativeTime(run.timestamp) }}</div>
          </button>
        </div>
      </section>
    </div>`,
};
