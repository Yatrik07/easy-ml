// Toasts, confirm dialog, modal and side drawer.
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from '../lib/vue.js';
import { closeDialog, dismissToast, ui } from '../store/ui.js';

export const ToastHost = {
  setup() {
    const icons = { success: 'check_circle', error: 'error', info: 'info' };
    return { ui, icons, dismissToast };
  },
  template: /* html */ `
    <div class="toast-host" aria-live="polite">
      <transition-group name="toast">
        <div v-for="t in ui.toasts" :key="t.id" class="toast" :class="'toast-' + t.type" role="status">
          <Icon :name="icons[t.type] || 'info'" />
          <div class="toast-message">{{ t.message }}</div>
          <button class="btn btn-ghost btn-icon" @click="dismissToast(t.id)" aria-label="Dismiss notification">
            <Icon name="close" />
          </button>
        </div>
      </transition-group>
    </div>`,
};

function useEscape(handler) {
  const onKey = event => { if (event.key === 'Escape') handler(); };
  onMounted(() => document.addEventListener('keydown', onKey));
  onBeforeUnmount(() => document.removeEventListener('keydown', onKey));
}

export const ConfirmDialog = {
  setup() {
    const confirmButton = ref(null);
    useEscape(() => { if (ui.dialog) closeDialog(false); });
    watch(() => ui.dialog, dialog => { if (dialog) nextTick(() => confirmButton.value && confirmButton.value.focus()); });
    return { ui, closeDialog, confirmButton };
  },
  template: /* html */ `
    <transition name="fade">
      <div v-if="ui.dialog" class="backdrop" @click.self="closeDialog(false)">
        <div class="modal" role="alertdialog" aria-modal="true" :aria-label="ui.dialog.title">
          <div class="modal-header"><h3>{{ ui.dialog.title }}</h3></div>
          <div class="modal-body"><p class="muted">{{ ui.dialog.message }}</p></div>
          <div class="modal-footer">
            <button class="btn btn-secondary" @click="closeDialog(false)">Cancel</button>
            <button ref="confirmButton" class="btn" :class="ui.dialog.danger ? 'btn-danger' : 'btn-primary'"
                    @click="closeDialog(true)">{{ ui.dialog.confirmLabel }}</button>
          </div>
        </div>
      </div>
    </transition>`,
};

export const ModalDialog = {
  props: { open: Boolean, title: String, wide: Boolean, xl: Boolean },
  emits: ['close'],
  setup(props, { emit }) {
    useEscape(() => { if (props.open) emit('close'); });
    return {};
  },
  template: /* html */ `
    <transition name="fade">
      <div v-if="open" class="backdrop" @click.self="$emit('close')">
        <div class="modal" :class="{ 'modal-wide': wide, 'modal-xl': xl }" role="dialog" aria-modal="true" :aria-label="title">
          <div class="modal-header">
            <h3>{{ title }}</h3>
            <button class="btn btn-ghost btn-icon" @click="$emit('close')" aria-label="Close"><Icon name="close" /></button>
          </div>
          <div class="modal-body"><slot /></div>
          <div class="modal-footer"><slot name="footer" /></div>
        </div>
      </div>
    </transition>`,
};

export const SideDrawer = {
  props: { open: Boolean, title: String, subtitle: String },
  emits: ['close'],
  setup(props, { emit }) {
    const closeButton = ref(null);
    let returnFocus = null;
    useEscape(() => { if (props.open) emit('close'); });
    watch(() => props.open, open => {
      if (open) {
        returnFocus = document.activeElement;
        nextTick(() => closeButton.value && closeButton.value.focus());
      } else if (returnFocus && returnFocus.focus) {
        returnFocus.focus();
      }
    });
    return { closeButton };
  },
  template: /* html */ `
    <transition name="fade"><div v-if="open" class="drawer-backdrop" @click="$emit('close')"></div></transition>
    <transition name="slide">
      <aside v-if="open" class="drawer" role="dialog" aria-modal="true" :aria-label="title">
        <div class="drawer-header">
          <div>
            <h2>{{ title }}</h2>
            <div v-if="subtitle" class="muted text-sm">{{ subtitle }}</div>
            <slot name="meta" />
          </div>
          <button ref="closeButton" class="btn btn-ghost btn-icon" @click="$emit('close')" aria-label="Close details">
            <Icon name="close" />
          </button>
        </div>
        <div class="drawer-body"><slot /></div>
      </aside>
    </transition>`,
};
