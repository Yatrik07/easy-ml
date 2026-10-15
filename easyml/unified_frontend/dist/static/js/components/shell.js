// App chrome: sidebar navigation, mobile top bar and the theme toggle.
import { computed } from '../lib/vue.js';
import { resolvedTheme, toggleTheme } from '../lib/theme.js';
import { navigate, ui } from '../store/ui.js';
import { session, taskLabel } from '../store/session.js';
import { dataReady, wizard } from '../store/wizard.js';
import { training } from '../store/training.js';

export const ThemeToggle = {
  setup() {
    const label = computed(() => (resolvedTheme.value === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'));
    return { resolvedTheme, toggleTheme, label };
  },
  template: /* html */ `
    <button class="nav-item" @click="toggleTheme" :aria-label="label" :title="label">
      <Icon :name="resolvedTheme === 'dark' ? 'light_mode' : 'dark_mode'" />
      <span>{{ resolvedTheme === 'dark' ? 'Light theme' : 'Dark theme' }}</span>
    </button>`,
};

export const AppSidebar = {
  setup() {
    const steps = computed(() => {
      const hasJob = training.status !== 'idle' && training.task === wizard.config.task;
      return [
        { view: 'upload', label: wizard.config.task === 'ImageClassification' ? 'Images' : 'Data', enabled: true },
        { view: 'config', label: 'Configure', enabled: dataReady.value },
        { view: 'models', label: 'Models', enabled: dataReady.value },
        { view: 'training', label: 'Train', enabled: hasJob },
        { view: 'results', label: 'Results', enabled: training.status === 'success' && hasJob },
      ];
    });
    const main = [
      { view: 'home', label: 'Home', icon: 'home' },
      { view: 'history', label: 'Run history', icon: 'history' },
      { view: 'prep', label: 'Data prep', icon: 'cleaning_services' },
      { view: 'schedules', label: 'Schedules', icon: 'event_available' },
    ];
    const footer = [
      { view: 'settings', label: 'Settings', icon: 'settings' },
      { view: 'about', label: 'About', icon: 'info' },
    ];
    const stepIndex = computed(() => steps.value.findIndex(s => s.view === ui.view));
    // Comparing runs is part of Run history.
    const isActive = view => ui.view === view || (view === 'history' && ui.view === 'compare');
    return { ui, session, wizard, steps, main, footer, navigate, taskLabel, stepIndex, training, isActive };
  },
  template: /* html */ `
    <div>
      <div v-if="ui.navOpen" class="sidebar-backdrop" @click="ui.navOpen = false"></div>
      <nav class="sidebar" :class="{ open: ui.navOpen }" aria-label="Main navigation">
        <div class="brand">
          <span class="brand-mark"><Icon name="auto_graph" /></span>
          Easy-ML Studio
          <small v-if="session.version">v{{ session.version }}</small>
        </div>

        <div class="nav-section">
          <button v-for="item in main" :key="item.view" class="nav-item" :class="{ active: isActive(item.view) }"
                  @click="navigate(item.view)" :aria-current="isActive(item.view) ? 'page' : null">
            <Icon :name="item.icon" /><span>{{ item.label }}</span>
          </button>
          <button class="nav-item" @click="navigate('home')" v-if="!wizard.config.task">
            <Icon name="add_circle_outline" /><span>New run</span>
          </button>
        </div>

        <div v-if="wizard.config.task" class="nav-section">
          <div class="nav-label">New run · {{ taskLabel(wizard.config.task) }}</div>
          <button v-for="(step, i) in steps" :key="step.view" class="nav-item"
                  :class="{ active: ui.view === step.view, done: stepIndex > i }"
                  :disabled="!step.enabled" @click="navigate(step.view)"
                  :aria-current="ui.view === step.view ? 'step' : null">
            <span class="step-index"><Icon v-if="stepIndex > i" name="check" size="sm" /><template v-else>{{ i + 1 }}</template></span>
            <span>{{ step.label }}</span>
            <span v-if="step.view === 'training' && training.status === 'running'" class="spinner ml-auto"></span>
          </button>
        </div>

        <div class="sidebar-footer">
          <button v-for="item in footer" :key="item.view" class="nav-item" :class="{ active: ui.view === item.view }"
                  @click="navigate(item.view)" :aria-current="ui.view === item.view ? 'page' : null">
            <Icon :name="item.icon" /><span>{{ item.label }}</span>
          </button>
          <ThemeToggle />
        </div>
      </nav>
    </div>`,
};

export const TopBar = {
  setup() { return { ui, session }; },
  template: /* html */ `
    <header class="topbar">
      <button class="btn btn-ghost btn-icon" @click="ui.navOpen = true" aria-label="Open navigation"><Icon name="menu" /></button>
      <div class="brand brand-compact">
        <span class="brand-mark"><Icon name="auto_graph" /></span>Easy-ML Studio
      </div>
    </header>`,
};
