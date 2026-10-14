import { computed, onMounted, ref } from '../lib/vue.js';
import { absoluteTime, formatScore, relativeTime } from '../lib/format.js';
import { closeRunDetail, deleteRun, loadRuns, openRunDetail, runs } from '../store/runs.js';
import { openInference } from '../store/inference.js';
import { confirmAction, navigate } from '../store/ui.js';
import { taskLabel } from '../store/session.js';

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'MLClassification', label: 'Classification' },
  { id: 'MLRegression', label: 'Regression' },
  { id: 'NLPTextClassification', label: 'Text' },
  { id: 'ImageClassification', label: 'Image' },
];

export default {
  setup() {
    const query = ref('');
    const filter = ref('all');
    onMounted(loadRuns);

    const filtered = computed(() => {
      const q = query.value.trim().toLowerCase();
      return runs.list.filter(run => {
        if (filter.value !== 'all' && run.task !== filter.value) return false;
        if (!q) return true;
        return [run.run_name, run.dataset_name, run.best_model_label, String(run.id)]
          .some(v => String(v || '').toLowerCase().includes(q));
      });
    });

    async function remove(run) {
      const ok = await confirmAction({
        title: 'Delete run #' + run.id + '?',
        message: '"' + run.run_name + '" and all of its trained models will be permanently deleted.',
        confirmLabel: 'Delete run',
        danger: true,
      });
      if (ok) deleteRun(run.id);
    }

    const detail = computed(() => runs.detail);
    const detailTitle = computed(() => (detail.value && detail.value.run_name) || (detail.value ? 'Run #' + detail.value.id : ''));

    return {
      FILTERS, query, filter, runs, filtered, remove, detail, detailTitle, openRunDetail, closeRunDetail,
      openInference, navigate, formatScore, relativeTime, absoluteTime, taskLabel,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader title="Run history" subtitle="Every model you have trained is saved here. Open a run to compare its models or make predictions.">
        <button class="btn btn-primary" @click="navigate('home')"><Icon name="add" />New run</button>
      </PageHeader>

      <div class="toolbar">
        <div class="search">
          <Icon name="search" />
          <input class="input" type="search" v-model="query" placeholder="Search runs, datasets or models" aria-label="Search runs" />
        </div>
        <div class="segmented" role="group" aria-label="Filter by task">
          <button v-for="f in FILTERS" :key="f.id" :class="{ active: filter === f.id }" @click="filter = f.id">{{ f.label }}</button>
        </div>
        <span class="spacer"></span>
        <span class="subtle text-sm">{{ filtered.length }} of {{ runs.list.length }} runs</span>
      </div>

      <div v-if="runs.loading && !runs.loaded" class="card card-body row muted"><Spinner /> Loading runs…</div>

      <div v-else-if="!runs.list.length" class="card">
        <EmptyState icon="model_training" title="No runs yet" text="Train your first model and it will appear here.">
          <button class="btn btn-primary" @click="navigate('home')"><Icon name="add" />Start a new run</button>
        </EmptyState>
      </div>

      <div v-else-if="!filtered.length" class="card">
        <EmptyState icon="search_off" title="No matching runs" text="Try a different search or task filter." />
      </div>

      <div v-else class="table-wrap">
        <table class="table">
          <thead>
            <tr>
              <th>Run</th>
              <th class="hide-md">Models</th>
              <th>Best score</th>
              <th class="hide-sm">Created</th>
              <th class="right"><span class="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="run in filtered" :key="run.id">
              <td>
                <div class="run-name">{{ run.run_name }}</div>
                <div class="run-sub"><span>#{{ run.id }}</span><TaskBadge :task="run.task" />
                  <span v-if="run.dataset_name" class="truncate dataset-name" :title="run.dataset_name"><Icon name="description" size="sm" />{{ run.dataset_name }}</span></div>
              </td>
              <td class="hide-md"><ModelChips :models="run.models" :max="1" /></td>
              <td>
                <div class="best-cell">
                  <span class="best-score">{{ formatScore(run.best_score, run.metric) }}</span>
                  <span class="subtle text-xs">{{ run.best_model_label || 'No score' }}</span>
                </div>
              </td>
              <td class="nowrap hide-sm" :title="absoluteTime(run.timestamp)">{{ relativeTime(run.timestamp) }}</td>
              <td>
                <div class="actions">
                  <button class="btn btn-secondary btn-sm" @click="openInference(run.id)" title="Make predictions with this run">
                    <Icon name="bolt" /><span class="btn-label">Predict</span>
                  </button>
                  <button class="btn btn-ghost btn-icon" @click="openRunDetail(run.id)" aria-label="Show details" title="Details">
                    <Icon name="insights" />
                  </button>
                  <button class="btn btn-ghost btn-icon" @click="remove(run)" aria-label="Delete run" title="Delete">
                    <Icon name="delete_outline" />
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <SideDrawer :open="!!detail" :title="detailTitle" @close="closeRunDetail">
        <template #meta>
          <div v-if="detail && detail.task" class="run-sub">
            <span>#{{ detail.id }}</span><TaskBadge :task="detail.task" />
            <span>{{ detail.dataset_name }}</span><span>· {{ absoluteTime(detail.timestamp) }}</span>
          </div>
        </template>
        <div v-if="runs.detailLoading" class="row muted"><Spinner /> Loading run…</div>
        <ResultsBoard v-else-if="detail && detail.metrics" :task="detail.task" :performance="detail.metrics" :run-id="detail.id">
          <template #actions>
            <button class="btn btn-primary btn-sm" @click="openInference(detail.id); closeRunDetail()"><Icon name="bolt" />Predict</button>
          </template>
        </ResultsBoard>
      </SideDrawer>
    </div>`,
};
