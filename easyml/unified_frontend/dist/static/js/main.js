// Entry point. Registers the shared components and mounts the app.
import { createApp } from './lib/vue.js';
import './lib/theme.js';
import App from './App.js';
import { loadSession } from './store/session.js';
import { EmptyState, Icon, ModelChips, PageHeader, ScoreValue, Spinner, StatTile, TaskBadge } from './components/ui.js';
import { ConfirmDialog, ModalDialog, SideDrawer, ToastHost } from './components/overlays.js';
import { CodeBlock, CodeTabs, CopyField } from './components/code.js';
import { ClassDistribution, ConfusionMatrix, ModelDetail, PipelineSummary, ResultsBoard } from './components/results.js';
import { AppSidebar, ThemeToggle, TopBar } from './components/shell.js';
import { CnnBuilder, FolderPicker } from './components/images.js';
import { ImportRunDialog, LineageBadge, RetrainDialog } from './components/runs.js';
import {
  AddStepDialog, MappingField, PreviewTable, ProfileTable, RecipesDialog, SaveRecipeDialog, StepForm, StepList, TrainDialog,
} from './components/prep.js';
import { ScheduleBadges, ScheduleDialog, ScheduleEventsDialog } from './components/schedules.js';
import { listenForScheduleEvents } from './store/schedules.js';
import {
  ContributionChart, ImportanceChart, ModelCharts, PrChart, ResidualChart, RocChart, ScatterChart, TopTerms,
} from './components/charts.js';

const app = createApp(App);

const components = {
  Icon, PageHeader, TaskBadge, ModelChips, ScoreValue, EmptyState, StatTile, Spinner,
  ToastHost, ConfirmDialog, ModalDialog, SideDrawer,
  CopyField, CodeBlock, CodeTabs,
  PipelineSummary, ConfusionMatrix, ModelDetail, ResultsBoard, ClassDistribution,
  AppSidebar, TopBar, ThemeToggle,
  FolderPicker, CnnBuilder,
  RetrainDialog, ImportRunDialog, LineageBadge,
  AddStepDialog, MappingField, PreviewTable, ProfileTable, RecipesDialog, SaveRecipeDialog, StepForm, StepList, TrainDialog,
  ScheduleDialog, ScheduleEventsDialog, ScheduleBadges,
  ModelCharts, RocChart, PrChart, ScatterChart, ResidualChart, ImportanceChart, TopTerms, ContributionChart,
};
for (const [name, component] of Object.entries(components)) app.component(name, component);

app.config.errorHandler = (err, instance, info) => {
  console.error('Easy-ML UI error:', info, err);
};

loadSession();
listenForScheduleEvents();
app.mount('#app');
window.__easymlBooted = true;
