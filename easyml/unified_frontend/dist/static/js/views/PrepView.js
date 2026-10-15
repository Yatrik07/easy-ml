import { onMounted, ref } from '../lib/vue.js';
import { formatNumber } from '../lib/format.js';
import { closeDataset, downloadCsv, loadRecipes, loadSpecs, prep, uploadPrepFile, useRecipe } from '../store/prep.js';

export default {
  setup() {
    const fileInput = ref(null);
    const dragging = ref(false);
    const tab = ref('data');
    const dialog = ref('');
    onMounted(() => { loadSpecs(); loadRecipes(); });

    function pick() { if (fileInput.value) fileInput.value.click(); }
    function onFile(event) { uploadPrepFile(event.target.files[0]); event.target.value = ''; }
    function onDrop(event) { dragging.value = false; uploadPrepFile(event.dataTransfer.files[0]); }

    return {
      prep, fileInput, dragging, tab, dialog, pick, onFile, onDrop, closeDataset, downloadCsv, useRecipe, formatNumber,
    };
  },
  template: /* html */ `
    <div class="page">
      <PageHeader eyebrow="Prepare data" title="Data prep"
                  subtitle="Clean and reshape a table step by step, with a live preview. Then train on it, download it, or save the steps as a recipe.">
        <template v-if="prep.dataset">
          <button class="btn btn-secondary" @click="dialog = 'recipes'"><Icon name="receipt_long" />Recipes</button>
          <button class="btn btn-secondary" @click="dialog = 'save'" :disabled="!prep.steps.length"><Icon name="save" />Save recipe</button>
          <button class="btn btn-secondary" @click="downloadCsv" :disabled="!!prep.busy">
            <Spinner v-if="prep.busy === 'download'" /><Icon v-else name="download" />Download CSV</button>
          <button class="btn btn-primary" @click="dialog = 'train'" :disabled="!prep.result || !!prep.busy"><Icon name="model_training" />Train with this data</button>
        </template>
      </PageHeader>

      <template v-if="!prep.dataset">
        <div class="grid-2">
          <div class="dropzone" :class="{ dragging }" role="button" tabindex="0" @click="pick" @keydown.enter="pick"
               @dragover.prevent="dragging = true" @dragleave="dragging = false" @drop.prevent="onDrop">
            <input ref="fileInput" type="file" accept=".csv,.xlsx,.xls" hidden @change="onFile" />
            <template v-if="prep.uploading"><Spinner /><strong>Uploading…</strong></template>
            <template v-else>
              <Icon name="upload_file" />
              <strong>Drop a file to clean, or click to browse</strong>
              <span class="text-sm">CSV, XLSX or XLS</span>
            </template>
          </div>
          <div class="card">
            <div class="card-header"><div><h3>Saved recipes</h3><div class="card-hint">Load one now; it runs on the file you upload.</div></div></div>
            <div class="card-body">
              <EmptyState v-if="!prep.recipes.length" icon="receipt_long" title="No recipes yet" text="Your saved step lists appear here." />
              <div v-else class="stack-xs">
                <div v-for="r in prep.recipes" :key="r.id" class="recipe-row">
                  <div class="truncate"><div class="strong truncate">{{ r.name }}</div>
                    <div class="subtle text-xs">{{ r.steps }} step{{ r.steps === 1 ? '' : 's' }}</div></div>
                  <button class="btn btn-secondary btn-sm" @click="useRecipe(r.id)"
                          :class="{ active: prep.recipe && prep.recipe.id === r.id }">{{ prep.recipe && prep.recipe.id === r.id ? 'Loaded' : 'Use' }}</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </template>

      <template v-else>
        <div class="card card-body prep-summary">
          <div class="row-nowrap">
            <Icon name="description" />
            <div class="truncate">
              <div class="strong truncate">{{ prep.dataset.fileName }}</div>
              <div class="subtle text-xs">{{ formatNumber(prep.dataset.rows) }} rows · {{ prep.dataset.cols }} columns
                <span v-if="prep.recipe"> · recipe "{{ prep.recipe.name }}"{{ prep.dirty ? ' (changed)' : '' }}</span></div>
            </div>
          </div>
          <div v-if="prep.result" class="prep-after">
            <span class="subtle text-xs">After {{ prep.steps.length }} step{{ prep.steps.length === 1 ? '' : 's' }}</span>
            <span class="strong">{{ formatNumber(prep.result.rows) }} × {{ prep.result.cols }}</span>
          </div>
          <button class="btn btn-ghost btn-sm" @click="closeDataset"><Icon name="swap_horiz" />Change file</button>
        </div>

        <div class="prep-layout">
          <section class="card">
            <div class="card-header">
              <div><h3>Steps</h3><div class="card-hint">Applied from top to bottom.</div></div>
              <button class="btn btn-primary btn-sm" @click="dialog = 'add'"><Icon name="add" />Add step</button>
            </div>
            <div class="card-body">
              <EmptyState v-if="!prep.steps.length" icon="playlist_add" title="No steps yet"
                          text="Add a step to start cleaning, for example remove an ID column or fill missing values." />
              <StepList v-else />
            </div>
          </section>

          <section class="card">
            <div class="card-header">
              <div class="tabs" role="tablist">
                <button role="tab" :aria-selected="tab === 'data'" :class="{ active: tab === 'data' }" @click="tab = 'data'">Data</button>
                <button role="tab" :aria-selected="tab === 'columns'" :class="{ active: tab === 'columns' }" @click="tab = 'columns'">Columns</button>
              </div>
              <span v-if="prep.loading" class="row subtle text-sm"><Spinner />Updating…</span>
            </div>
            <div class="card-body stack-sm">
              <div v-if="prep.error" class="alert alert-danger"><Icon name="error_outline" /><div>{{ prep.error }}</div></div>
              <template v-if="prep.result">
                <p v-if="prep.result.sampled" class="subtle text-xs">The preview uses the first {{ formatNumber(prep.result.sample_rows) }} rows;
                  downloading or training applies the steps to all {{ formatNumber(prep.dataset.rows) }}.</p>
                <PreviewTable v-if="tab === 'data'" :result="prep.result" />
                <ProfileTable v-else :profile="prep.result.profile" />
              </template>
              <div v-else-if="prep.loading" class="row muted"><Spinner /> Reading the file…</div>
            </div>
          </section>
        </div>
      </template>

      <AddStepDialog :open="dialog === 'add'" @close="dialog = ''" />
      <RecipesDialog :open="dialog === 'recipes'" @close="dialog = ''" />
      <SaveRecipeDialog :open="dialog === 'save'" @close="dialog = ''" />
      <TrainDialog :open="dialog === 'train'" @close="dialog = ''" />
    </div>`,
};
