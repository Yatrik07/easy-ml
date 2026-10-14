// UI state: current view, mobile navigation, toasts and the confirm dialog.
// Stores are plain reactive objects; import the object and read its fields (never destructure).
import { reactive } from '../lib/vue.js';

export const ui = reactive({
  view: 'home',
  navOpen: false,
  toasts: [],
  dialog: null, // { title, message, confirmLabel, danger, resolve }
});

const listeners = [];

// Views register a callback to refresh when they are opened (e.g. Settings re-reads disk usage).
export function onNavigate(fn) { listeners.push(fn); }

export function navigate(view) {
  ui.view = view;
  ui.navOpen = false;
  listeners.forEach(fn => fn(view));
  window.scrollTo({ top: 0 });
  const main = document.querySelector('.main');
  if (main) main.scrollTop = 0;
}

let toastId = 0;

export function toast(message, type = 'info', duration = 6000) {
  const id = ++toastId;
  ui.toasts.push({ id, message, type });
  setTimeout(() => dismissToast(id), duration);
}
export const toastError = message => toast(message, 'error', 9000);
export const toastSuccess = message => toast(message, 'success', 5000);

export function dismissToast(id) {
  ui.toasts = ui.toasts.filter(t => t.id !== id);
}

// Promise-based replacement for window.confirm().
export function confirmAction({ title, message, confirmLabel = 'Confirm', danger = false }) {
  return new Promise(resolve => {
    ui.dialog = { title, message, confirmLabel, danger, resolve };
  });
}

export function closeDialog(result) {
  if (ui.dialog) ui.dialog.resolve(result);
  ui.dialog = null;
}
