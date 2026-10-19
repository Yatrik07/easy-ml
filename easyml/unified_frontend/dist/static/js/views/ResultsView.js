import { computed } from '../lib/vue.js';
import { metricFor, rankPerformance } from '../lib/format.js';
import { modelLabel, taskLabel } from '../store/session.js';
import { navigate } from '../store/ui.js';
import { training } from '../store/training.js';
import { openInference } from '../store/inference.js';
import { openFinalDialog } from '../store/final.js';

export default {
  setup() {
    const result = computed(() => training.result || {});
    const performance = computed(() => result.value.performance || {});
    const best = computed(() => rankPerformance(training.task, performance.value, modelLabel)[0]);
    const metric = computed(() => metricFor(training.task));
    const eyebrow = computed(() => {
      if (training.kind === 'retrain') return 'Retrained from run #' + training.parentRunId + ' · ' + taskLabel(training.task);
      if (training.kind === 'final') return 'Final model of run #' + training.parentRunId + ' · ' + taskLabel(training.task);
      if (training.kind === 'schedule') return 'Schedule ' + training.label + ' · ' + taskLabel(training.task);
      return 'Step 5 · ' + taskLabel(training.task);
    });
    // A final model can be trained from an evaluated table or text run (not from a final model itself).
    const canFinalize = computed(() => result.value.run_id > 0 && training.kind !== 'final'
      && training.task !== 'ImageClassification');
    return { training, result, performance, best, metric, navigate, openInference, openFinalDialog, eyebrow, canFinalize };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader :eyebrow="eyebrow" title="Results"
                  :subtitle="best ? 'Best model: ' + best.label + '. Every model is saved in run #' + result.run_id + '.' : ''">
        <button class="btn btn-secondary" @click="navigate('home')"><Icon name="add" />New run</button>
        <a v-if="result.run_id > 0" class="btn btn-secondary" :href="'/runs/' + result.run_id + '/report'" download
           title="One HTML file with the scores, charts and settings, to send to anyone"><Icon name="description" />Download report</a>
        <button v-if="canFinalize" class="btn btn-secondary" @click="openFinalDialog(result.run_id)"
                title="Train the chosen model again on all the data (training and test rows), for use"><Icon name="verified" />Train final model</button>
        <button v-if="result.run_id > 0" class="btn btn-primary" @click="openInference(result.run_id)"><Icon name="bolt" />Make predictions</button>
      </PageHeader>

      <div v-if="!Object.keys(performance).length" class="card">
        <EmptyState icon="insights" title="No results yet" text="Train a model to see its results here." />
      </div>
      <ResultsBoard v-else :task="training.task" :performance="performance" :summary="result.pipeline_summary" :run-id="result.run_id" />
      <FinalModelDialog />
    </div>`,
};
