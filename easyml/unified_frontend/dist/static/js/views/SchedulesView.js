import { onMounted, onUnmounted } from '../lib/vue.js';
import { absoluteTime, formatScore, relativeTime, untilTime } from '../lib/format.js';
import { postJSON } from '../lib/api.js';
import { openRunDetail } from '../store/runs.js';
import { training } from '../store/training.js';
import { navigate, toastError } from '../store/ui.js';
import {
  deleteSchedule, loadSchedules, openEditDialog, openEvents, openScheduleDialog, runScheduleNow, schedules, toggleSchedule,
} from '../store/schedules.js';
import { decisionInfo } from '../components/schedules.js';

export default {
  setup() {
    let timer = null;
    onMounted(() => {
      loadSchedules();
      timer = setInterval(loadSchedules, 10000); // running state and "next run" stay current
    });
    onUnmounted(() => clearInterval(timer));

    function openRun(id) {
      navigate('history');
      openRunDetail(id);
    }
    async function stop(schedule) {
      try {
        await postJSON('/stop_training', { job_id: schedule.job_id }, 'Could not stop the run');
      } catch (e) {
        toastError(e.message);
      }
    }
    function score(run) {
      return run && run.exists && run.score !== null ? formatScore(run.score, run.metric) : '';
    }
    return {
      schedules, training, openScheduleDialog, openEditDialog, openEvents, runScheduleNow, deleteSchedule, toggleSchedule,
      decisionInfo, openRun, stop, score, relativeTime, absoluteTime, untilTime,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader eyebrow="Keep models fresh" title="Schedules"
                  subtitle="Retrain a run automatically on new data. A new model replaces the one in use only when it scores better.">
        <button class="btn btn-primary" @click="openScheduleDialog()"><Icon name="add" />New schedule</button>
      </PageHeader>

      <div class="alert" :class="schedules.scheduler.studio ? 'alert-info' : 'alert-warning'">
        <Icon name="info" />
        <div v-if="schedules.scheduler.studio">Schedules run while Easy-ML Studio is open (checked every
          {{ schedules.scheduler.poll_seconds }} seconds). To run them when the studio is closed, start
          <code>easy-ml schedule worker</code>, or run <code>easy-ml schedule run-due</code> from Windows Task Scheduler or cron.</div>
        <div v-else>This studio is not running schedules (it was started with <code>--no-scheduler</code> or <code>EASYML_SCHEDULER=0</code>).
          Run them with <code>easy-ml schedule worker</code>.</div>
      </div>

      <div v-if="!schedules.loaded" class="row muted"><Spinner /> Loading schedules…</div>
      <EmptyState v-else-if="!schedules.list.length" icon="event_available" title="No schedules yet"
                  text="Pick a run, point it at the file or folder where new data arrives, and choose when to retrain.">
        <button class="btn btn-primary" @click="openScheduleDialog()"><Icon name="add" />New schedule</button>
      </EmptyState>

      <div v-else class="schedule-list">
        <article v-for="s in schedules.list" :key="s.id" class="card schedule-card" :class="{ off: !s.enabled }">
          <div class="card-header">
            <div class="truncate">
              <h3 class="truncate">{{ s.name }}</h3>
              <div class="card-hint">{{ s.task_label }} · {{ s.when }}</div>
            </div>
            <label class="row-nowrap text-sm">
              <input type="checkbox" :checked="s.enabled" @change="toggleSchedule(s)" :aria-label="'Schedule ' + s.name + ' switched on'" />
              {{ s.enabled ? 'On' : 'Off' }}</label>
          </div>
          <div class="card-body stack-sm">
            <dl class="kv">
              <dt>Model in use</dt>
              <dd class="schedule-fact">
                <button v-if="s.champion.exists" class="btn btn-ghost btn-sm schedule-run" @click="openRun(s.champion.id)">
                  #{{ s.champion.id }} {{ s.champion.name }}</button>
                <span v-else class="field-error">Run #{{ s.champion.id }} was deleted</span>
                <span v-if="score(s.champion)" class="subtle">{{ s.champion.best_model }} {{ score(s.champion) }}</span>
              </dd>
              <dt>Next run</dt>
              <dd class="schedule-fact">
                <span v-if="s.running" class="row-nowrap"><Spinner />Training now</span>
                <span v-else-if="!s.enabled" class="subtle">Off</span>
                <span v-else-if="s.trigger === 'new_data'">When new data arrives</span>
                <span v-else :title="absoluteTime(s.next_run_at)">{{ untilTime(s.next_run_at) }}</span>
              </dd>
              <dt>Last result</dt>
              <dd class="schedule-fact">
                <span v-if="s.last_status" class="badge" :class="decisionInfo(s.last_status).badge">
                  <Icon :name="decisionInfo(s.last_status).icon" />{{ decisionInfo(s.last_status).label }}</span>
                <span v-else class="subtle">Not run yet</span>
                <span v-if="s.last_run_at" class="subtle text-xs" :title="absoluteTime(s.last_run_at)">{{ relativeTime(s.last_run_at) }}</span>
              </dd>
              <dt>New data</dt>
              <dd class="truncate" :title="s.source_path"><code>{{ s.source_path }}</code></dd>
            </dl>
            <p v-if="s.last_message" class="subtle text-xs schedule-message">{{ s.last_message }}</p>
          </div>
          <div class="card-footer row">
            <button v-if="s.running" class="btn btn-danger-outline btn-sm" @click="stop(s)" :disabled="!s.job_id"><Icon name="stop" />Stop</button>
            <button v-else class="btn btn-secondary btn-sm" @click="runScheduleNow(s)" :disabled="training.status === 'running'"
                    title="Retrain now on the current data"><Icon name="play_arrow" />Run now</button>
            <button class="btn btn-ghost btn-sm" @click="openEvents(s)"><Icon name="history" />History</button>
            <button class="btn btn-ghost btn-sm" @click="openEditDialog(s)"><Icon name="edit" />Edit</button>
            <button class="btn btn-ghost btn-sm" @click="deleteSchedule(s)" :disabled="s.running"><Icon name="delete_outline" />Delete</button>
          </div>
        </article>
      </div>

      <ScheduleDialog />
      <ScheduleEventsDialog />
    </div>`,
};
