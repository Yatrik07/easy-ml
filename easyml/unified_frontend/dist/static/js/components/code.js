// Copyable commands and code snippets.
import { ref } from '../lib/vue.js';
import { copyText } from '../lib/clipboard.js';
import { toastError } from '../store/ui.js';

function useCopy() {
  const copied = ref(false);
  async function copy(text) {
    if (await copyText(text)) {
      copied.value = true;
      setTimeout(() => { copied.value = false; }, 1500);
    } else {
      toastError('Could not copy. Select the text and copy it manually.');
    }
  }
  return { copied, copy };
}

export const CopyField = {
  props: { text: { type: String, required: true }, label: { type: String, default: 'Copy' } },
  setup() { return useCopy(); },
  template: /* html */ `
    <div class="copy-field">
      <code :title="text">{{ text }}</code>
      <button class="btn btn-ghost btn-icon" @click="copy(text)" :aria-label="label" :title="copied ? 'Copied' : label">
        <Icon :name="copied ? 'check' : 'content_copy'" />
      </button>
    </div>`,
};

export const CodeBlock = {
  props: { code: { type: String, required: true } },
  setup() { return useCopy(); },
  template: /* html */ `
    <div class="code-block">{{ code }}<button class="btn btn-ghost btn-icon copy-btn" @click="copy(code)"
        :aria-label="copied ? 'Copied' : 'Copy code'" :title="copied ? 'Copied' : 'Copy'"><Icon :name="copied ? 'check' : 'content_copy'" /></button></div>`,
};

export const CodeTabs = {
  props: { tabs: { type: Array, required: true } }, // [{ id, label, code }]
  setup(props) {
    const active = ref(props.tabs.length ? props.tabs[0].id : null);
    return { active };
  },
  template: /* html */ `
    <div>
      <div class="tabs" role="tablist">
        <button v-for="tab in tabs" :key="tab.id" role="tab" :aria-selected="active === tab.id"
                :class="{ active: active === tab.id }" @click="active = tab.id">{{ tab.label }}</button>
      </div>
      <template v-for="tab in tabs" :key="tab.id">
        <CodeBlock v-if="active === tab.id" :code="tab.code" />
      </template>
    </div>`,
};
