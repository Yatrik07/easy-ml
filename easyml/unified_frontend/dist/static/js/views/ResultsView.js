import { computed } from '../lib/vue.js';
import { metricFor, rankPerformance } from '../lib/format.js';
import { modelLabel, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import { training } from '../store/training.js';
import { openInference } from '../store/inference.js';

export default {
  setup() {
    const result = computed(() => training.result || {});
    const performance = computed(() => result.value.performance || {});
    const best = computed(() => rankPerformance(training.task, performance.value, modelLabel)[0]);
    const metric = computed(() => metricFor(training.task));
    const eyebrow = computed(() => (training.kind === 'retrain'
      ? 'Retrained from run #' + training.parentRunId + ' · ' + taskLabel(training.task)
      : 'Step 5 · ' + taskLabel(training.task)));
    return { training, result, performance, best, metric, navigate, openInference, eyebrow };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader :eyebrow="eyebrow" title="Results"
                  :subtitle="best ? 'Best model: ' + best.label + '. Every model is saved in run #' + result.run_id + '.' : ''">
        <button class="btn btn-secondary" @click="navigate('home')"><Icon name="add" />New run</button>
        <button v-if="result.run_id > 0" class="btn btn-primary" @click="openInference(result.run_id)"><Icon name="bolt" />Make predictions</button>
      </PageHeader>

      <div v-if="!Object.keys(performance).length" class="card">
        <EmptyState icon="insights" title="No results yet" text="Train a model to see its results here." />
      </div>
      <ResultsBoard v-else :task="training.task" :performance="performance" :summary="result.pipeline_summary" :run-id="result.run_id" />
    </div>`,
};
