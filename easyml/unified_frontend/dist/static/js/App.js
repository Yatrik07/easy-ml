import { computed } from './lib/vue.js';
import { ui } from './store/ui.js';
import HomeView from './views/HomeView.js';
import UploadView from './views/UploadView.js';
import ConfigView from './views/ConfigView.js';
import ModelsView from './views/ModelsView.js';
import TrainingView from './views/TrainingView.js';
import ResultsView from './views/ResultsView.js';
import InferenceView from './views/InferenceView.js';
import HistoryView from './views/HistoryView.js';
import SettingsView from './views/SettingsView.js';
import AboutView from './views/AboutView.js';

const VIEWS = {
  home: HomeView,
  upload: UploadView,
  config: ConfigView,
  models: ModelsView,
  training: TrainingView,
  results: ResultsView,
  inference: InferenceView,
  history: HistoryView,
  settings: SettingsView,
  about: AboutView,
};

export default {
  setup() {
    const current = computed(() => VIEWS[ui.view] || HomeView);
    return { ui, current };
  },
  template: /* html */ `
    <div class="app">
      <AppSidebar />
      <div class="main">
        <TopBar />
        <main id="main-content">
          <component :is="current" :key="ui.view" />
        </main>
      </div>
      <ToastHost />
      <ConfirmDialog />
    </div>`,
};
