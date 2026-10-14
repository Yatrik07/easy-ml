// Image-classification helpers: the server folder picker and the custom CNN builder.
import { computed } from '../lib/vue.js';
import { formatNumber } from '../lib/format.js';
import {
  addLayer, chooseFolder, LAYER_TYPES, moveLayer, removeLayer, validateCnn, wizard,
} from '../store/wizard.js';

export const FolderPicker = {
  setup() {
    const picker = computed(() => wizard.folderPicker);
    return { wizard, picker, chooseFolder };
  },
  template: /* html */ `
    <ModalDialog :open="picker.open" title="Choose an image folder" @close="picker.open = false">
      <p class="muted text-sm mb-3">Folders inside your Easy-ML data directory. Each needs one sub-folder per class.</p>
      <div v-if="picker.loading" class="row muted"><Spinner /> Loading folders…</div>
      <EmptyState v-else-if="!picker.folders.length" icon="folder_off" title="No folders found"
                  text="Copy an image dataset into the Easy-ML data directory (shown in Settings), or type a folder path instead." />
      <div v-else class="stack-sm" role="listbox">
        <label v-for="f in picker.folders" :key="f.path" class="check" :class="{ checked: picker.selected === f.path }">
          <input type="radio" name="folder" :value="f.path" v-model="picker.selected" />
          <div>
            <div class="check-title">{{ f.name }}</div>
            <div class="check-hint">{{ f.num_classes }} class folder{{ f.num_classes === 1 ? '' : 's' }}</div>
          </div>
        </label>
      </div>
      <template #footer>
        <button class="btn btn-secondary" @click="picker.open = false">Cancel</button>
        <button class="btn btn-primary" :disabled="!picker.selected" @click="chooseFolder">Use folder</button>
      </template>
    </ModalDialog>`,
};

export const CnnBuilder = {
  setup() {
    const layerCount = computed(() => wizard.layers.length);
    return { wizard, LAYER_TYPES, addLayer, removeLayer, moveLayer, validateCnn, layerCount, formatNumber };
  },
  template: /* html */ `
    <div class="card">
      <div class="card-header">
        <div>
          <h3>Custom CNN architecture</h3>
          <div class="card-hint">The network starts from the image input. The last Linear layer is resized to your number of classes automatically.</div>
        </div>
        <button class="btn btn-secondary btn-sm" @click="validateCnn"><Icon name="fact_check" />Validate</button>
      </div>
      <div class="card-body stack">
        <div class="layer-list">
          <div v-for="(layer, idx) in wizard.layers" :key="idx" class="layer-row">
            <span class="layer-type">{{ layer.type }}</span>
            <template v-if="layer.type === 'Conv2d'">
              <label class="sr-only" :for="'oc' + idx">Output channels</label>
              <input :id="'oc' + idx" class="input" type="number" min="1" v-model.number="layer.out_channels" title="Output channels" placeholder="channels" />
              <input class="input" type="number" min="1" v-model.number="layer.kernel_size" title="Kernel size" placeholder="kernel" aria-label="Kernel size" />
              <input class="input" type="number" min="0" v-model.number="layer.padding" title="Padding" placeholder="padding" aria-label="Padding" />
            </template>
            <input v-if="layer.type === 'MaxPool2d'" class="input" type="number" min="1" v-model.number="layer.kernel_size" title="Pool size" aria-label="Pool size" />
            <input v-if="layer.type === 'Dropout'" class="input" type="number" step="0.1" min="0" max="0.9" v-model.number="layer.p" title="Drop probability" aria-label="Drop probability" />
            <template v-if="layer.type === 'Linear'">
              <input v-if="idx !== layerCount - 1" class="input" type="number" min="1" v-model.number="layer.out_features" title="Output features" aria-label="Output features" />
              <span v-else class="subtle text-sm">output = number of classes</span>
            </template>
            <div class="layer-actions">
              <button class="btn btn-ghost btn-icon" :disabled="idx === 0" @click="moveLayer(idx, -1)" aria-label="Move layer up"><Icon name="arrow_upward" /></button>
              <button class="btn btn-ghost btn-icon" :disabled="idx === layerCount - 1" @click="moveLayer(idx, 1)" aria-label="Move layer down"><Icon name="arrow_downward" /></button>
              <button class="btn btn-ghost btn-icon" @click="removeLayer(idx)" aria-label="Remove layer"><Icon name="delete_outline" /></button>
            </div>
          </div>
        </div>
        <div class="row">
          <span class="subtle text-sm">Add layer:</span>
          <button v-for="type in LAYER_TYPES" :key="type" class="btn btn-secondary btn-sm" @click="addLayer(type)">+ {{ type }}</button>
        </div>
        <div v-if="wizard.cnnValidation && wizard.cnnValidation.valid" class="alert alert-success">
          <Icon name="check_circle" />
          <div>Valid architecture · output {{ wizard.cnnValidation.output_shape.join(' × ') }} ·
            {{ formatNumber(wizard.cnnValidation.total_params) }} parameters ({{ formatNumber(wizard.cnnValidation.trainable_params) }} trainable)</div>
        </div>
        <div v-else-if="wizard.cnnValidation" class="alert alert-danger">
          <Icon name="error_outline" /><div>{{ wizard.cnnValidation.error }}</div>
        </div>
      </div>
    </div>`,
};
