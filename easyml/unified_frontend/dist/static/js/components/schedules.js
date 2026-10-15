// Scheduled retraining: the create/edit dialog, a schedule's history and the Run history badges.
import { computed, watch } from '../lib/vue.js';
import { formatScore, relativeTime, absoluteTime } from '../lib/format.js';
import { openRunDetail, runs } from '../store/runs.js';
import { navigate } from '../store/ui.js';
import {
  TRIGGERS, WEEKDAYS, checkSource, closeEvents, closeScheduleDialog, marginInPoints, runFor, saveSchedule, schedules,
} from '../store/schedules.js';

export const DECISIONS = {
  promoted: { label: 'New model in use', badge: 'badge-success', icon: 'trending_up' },
  'kept champion': { label: 'Kept current model', badge: 'badge-accent', icon: 'verified_user' },
  failed: { label: 'Failed', badge: 'badge-danger', icon: 'error_outline' },
  'no new data': { label: 'No new data', badge: '', icon: 'hourglass_empty' },
};

export function decisionInfo(decision) {
  return DECISIONS[decision] || { label: decision || 'Not run yet', badge: '', icon: 'schedule' };
}

export const ScheduleDialog = {
  setup() {
    const run = computed(() => runFor(schedules.dialog.form.run_id));
    const task = computed(() => (run.value ? run.value.task : (schedules.dialog.editing ? schedules.dialog.editing.task : '')));
    const isImage = computed(() => task.value === 'ImageClassification');
    const points = computed(() => marginInPoints(task.value));
    const check = computed(() => schedules.dialog.check);
    const checkClass = computed(() => {
      if (!check.value) return '';
      if (!check.value.ok) return 'alert-danger';
      return check.value.missing_columns && check.value.missing_columns.length ? 'alert-warning' : 'alert-success';
    });
    const triggerHint = computed(() => (schedules.dialog.form.trigger === 'new_data'
      ? 'Checked every ' + schedules.scheduler.poll_seconds + ' seconds. A file counts once it has not changed for 10 seconds, so half-copied files are skipped.'
      : 'Skipped when the data has not changed since the last run. "Run now" always retrains.'));
    const ready = computed(() => {
      const form = schedules.dialog.form;
      if (!form.run_id || !form.name.trim() || !form.source_path.trim() || schedules.dialog.saving) return false;
      if (form.trigger === 'interval' && !(Number(form.interval_hours) >= 0.1)) return false;
      if ((form.trigger === 'daily' || form.trigger === 'weekly') && !form.at_time) return false;
      return !check.value || check.value.ok;
    });
    watch(() => [schedules.dialog.form.run_id, schedules.dialog.form.source_path, schedules.dialog.form.file_pattern],
      () => { if (schedules.dialog.open) checkSource(400); });
    watch(() => schedules.dialog.form.run_id, id => {
      const chosen = runFor(id);
      if (!schedules.dialog.editing && chosen && !schedules.dialog.form.name) {
        schedules.dialog.form.name = (chosen.run_name + '-schedule').slice(0, 80);
      }
    });
    return {
      schedules, runs, run, isImage, points, check, checkClass, triggerHint, ready, TRIGGERS, WEEKDAYS,
      closeScheduleDialog, saveSchedule,
    };
  },
  template: /* html */ `
    <ModalDialog :open="schedules.dialog.open" :title="schedules.dialog.editing ? 'Edit schedule' : 'Schedule retraining'"
                 @close="closeScheduleDialog">
      <div class="stack">
        <p class="muted text-sm">Retrains a run on new data with the same settings. The new model replaces the one in use
          only if it scores better on data neither has trained on; otherwise the current model stays and the new run is kept
          in Run history.</p>
        <div class="field">
          <label for="sched-run">Run to keep fresh</label>
          <select id="sched-run" class="select" v-model="schedules.dialog.form.run_id" :disabled="!!schedules.dialog.editing">
            <option value="">Choose a run…</option>
            <option v-for="r in runs.list" :key="r.id" :value="String(r.id)">#{{ r.id }} {{ r.run_name }} · {{ r.task_label }}</option>
          </select>
          <div v-if="schedules.dialog.editing" class="field-hint">Model in use now: run #{{ schedules.dialog.editing.champion.id }}.</div>
        </div>
        <div class="field">
          <label for="sched-name">Name</label>
          <input id="sched-name" class="input" v-model="schedules.dialog.form.name" maxlength="80" />
        </div>
        <div class="field">
          <label for="sched-source">Where the new data comes from</label>
          <input id="sched-source" class="input" v-model="schedules.dialog.form.source_path" spellcheck="false"
                 :placeholder="isImage ? 'C:/data/photos' : 'C:/exports/customers.csv, or a folder'" />
          <div class="field-hint">{{ isImage
            ? 'A folder with one sub-folder per class. Add images to it; each run trains on the whole folder.'
            : 'A file you replace with newer data, or a folder where new files appear (the newest one is used). The data is copied before training.' }}</div>
        </div>
        <div v-if="check && check.kind === 'folder'" class="field">
          <label for="sched-pattern">File names to use</label>
          <input id="sched-pattern" class="input" v-model="schedules.dialog.form.file_pattern" placeholder="*.csv;*.xlsx;*.xls"
                 spellcheck="false" />
        </div>
        <div v-if="schedules.dialog.checking" class="row muted text-sm"><Spinner /> Checking the data…</div>
        <div v-else-if="check" class="alert" :class="checkClass">
          <Icon :name="check.ok ? (checkClass === 'alert-success' ? 'check_circle' : 'info') : 'error_outline'" />
          <div>{{ check.message }}</div>
        </div>

        <div class="field">
          <span class="field-label">When to retrain</span>
          <div class="segmented" role="group" aria-label="When to retrain">
            <button v-for="t in TRIGGERS" :key="t.id" type="button" :class="{ active: schedules.dialog.form.trigger === t.id }"
                    @click="schedules.dialog.form.trigger = t.id">{{ t.label }}</button>
          </div>
          <div v-if="schedules.dialog.form.trigger !== 'new_data'" class="form-grid">
            <div v-if="schedules.dialog.form.trigger === 'interval'" class="field">
              <label for="sched-hours">Every (hours)</label>
              <input id="sched-hours" type="number" min="0.1" step="0.5" class="input" v-model="schedules.dialog.form.interval_hours" />
            </div>
            <div v-if="schedules.dialog.form.trigger === 'weekly'" class="field">
              <label for="sched-day">Day</label>
              <select id="sched-day" class="select" v-model="schedules.dialog.form.weekday">
                <option v-for="(d, i) in WEEKDAYS" :key="d" :value="i">{{ d }}</option>
              </select>
            </div>
            <div v-if="schedules.dialog.form.trigger !== 'interval'" class="field">
              <label for="sched-time">Time</label>
              <input id="sched-time" type="time" class="input" v-model="schedules.dialog.form.at_time" />
            </div>
          </div>
          <div class="field-hint">{{ triggerHint }}</div>
        </div>

        <div v-if="!isImage" class="field">
          <label for="sched-recipe">Clean the new data first</label>
          <select id="sched-recipe" class="select" v-model="schedules.dialog.form.recipe">
            <option value="">With the run's own data-prep recipe, if it has one</option>
            <option value="none">No cleaning, use the data as it is</option>
            <option v-for="r in schedules.dialog.recipes" :key="r.id" :value="String(r.id)">Recipe: {{ r.name }} ({{ r.steps }} steps)</option>
          </select>
        </div>
        <div class="field">
          <label for="sched-margin">Use the new model only if it is better by at least</label>
          <div class="row-nowrap">
            <input id="sched-margin" type="number" min="0" :max="points ? 100 : 1" :step="points ? 0.5 : 0.01"
                   class="input input-number" v-model="schedules.dialog.form.margin" />
            <span class="muted text-sm">{{ points ? 'accuracy points' : 'R²' }}</span>
          </div>
          <div class="field-hint">0 switches whenever the new model scores at least as well as the current one.</div>
        </div>
        <label class="row text-sm"><input type="checkbox" v-model="schedules.dialog.form.enabled" />Switched on</label>
      </div>
      <template #footer>
        <button class="btn btn-secondary" @click="closeScheduleDialog">Cancel</button>
        <button class="btn btn-primary" :disabled="!ready" @click="saveSchedule">
          <Spinner v-if="schedules.dialog.saving" /><Icon v-else name="event_available" />{{ schedules.dialog.editing ? 'Save' : 'Create schedule' }}
        </button>
      </template>
    </ModalDialog>`,
};

export const ScheduleEventsDialog = {
  setup() {
    function openRun(id) {
      closeEvents();
      navigate('history');
      openRunDetail(id);
    }
    function score(event, value) {
      return value === null || value === undefined ? 'n/a' : formatScore(value, { kind: event.metric === 'R²' ? 'decimal' : 'percent' });
    }
    return { schedules, closeEvents, decisionInfo, openRun, score, relativeTime, absoluteTime };
  },
  template: /* html */ `
    <ModalDialog :open="schedules.events.open" wide
                 :title="schedules.events.schedule ? 'History of ' + schedules.events.schedule.name : 'History'" @close="closeEvents">
      <div v-if="schedules.events.loading" class="row muted"><Spinner /> Loading…</div>
      <EmptyState v-else-if="!schedules.events.list.length" icon="history" title="Nothing yet"
                  text="Each time the schedule trains a model, the result appears here." />
      <div v-else class="table-wrap">
        <table class="table table-compact">
          <thead><tr><th>When</th><th>Result</th><th>New run</th><th class="right">New</th><th class="right">In use</th><th class="hide-sm">Data</th></tr></thead>
          <tbody>
            <template v-for="e in schedules.events.list" :key="e.id">
              <tr>
                <td class="nowrap" :title="absoluteTime(e.started_at)">{{ relativeTime(e.started_at) }}
                  <div class="subtle text-xs">{{ e.trigger }}</div></td>
                <td><span class="badge" :class="decisionInfo(e.decision).badge"><Icon :name="decisionInfo(e.decision).icon" />{{ decisionInfo(e.decision).label }}</span></td>
                <td><button v-if="e.challenger_run_id" class="btn btn-ghost btn-sm" @click="openRun(e.challenger_run_id)">#{{ e.challenger_run_id }}</button>
                  <span v-else class="subtle">-</span></td>
                <td class="right num">{{ score(e, e.challenger_score) }}</td>
                <td class="right num">{{ score(e, e.champion_score) }}<div v-if="e.champion_before" class="subtle text-xs">run #{{ e.champion_before }}</div></td>
                <td class="hide-sm"><span class="truncate event-file" :title="e.data_file">{{ e.data_file || '-' }}</span>
                  <div v-if="e.compared_on" class="subtle text-xs">compared on {{ e.compared_on }}</div></td>
              </tr>
              <tr v-if="e.message" class="event-message"><td colspan="6" class="subtle text-xs">{{ e.message }}</td></tr>
            </template>
          </tbody>
        </table>
      </div>
      <template #footer>
        <button class="btn btn-secondary" @click="closeEvents">Close</button>
      </template>
    </ModalDialog>`,
};

// Run history: which schedule uses a run, and scheduled runs that did not replace the model in use.
export const ScheduleBadges = {
  props: { run: Object },
  setup() {
    return { navigate };
  },
  template: /* html */ `
    <template v-if="run">
      <button v-for="name in run.schedule_champion_of || []" :key="name" class="badge badge-success lineage"
              @click.stop="navigate('schedules')" :title="'The model schedule ' + name + ' uses now'">
        <Icon name="event_available" />In use by {{ name }}</button>
      <span v-if="run.schedule_decision === 'kept champion'" class="badge lineage"
            title="A schedule trained this run but kept the model it already used, which scored better">
        <Icon name="verified_user" />Not promoted</span>
    </template>`,
};
