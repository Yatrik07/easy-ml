import { onMounted, ref } from '../lib/vue.js';
import { getJSON } from '../lib/api.js';
import { initials } from '../lib/format.js';
import { session } from '../store/session.js';

const LINK_ICONS = { pypi: 'inventory_2', github: 'code', docs: 'menu_book', issues: 'bug_report' };

export default {
  setup() {
    const credits = ref(null);
    const error = ref('');
    onMounted(async () => {
      try { credits.value = await getJSON('/settings/credits', 'Could not load project information'); }
      catch (e) { error.value = e.message; }
    });
    return { credits, error, session, initials, LINK_ICONS };
  },
  template: /* html */ `
    <div class="page">
      <div v-if="error" class="alert alert-danger"><Icon name="error_outline" /><div>{{ error }}</div></div>

      <section class="card card-body stack" v-if="credits">
        <div class="about-hero">
          <div class="about-logo"><Icon name="auto_graph" /></div>
          <div class="stack-sm">
            <h1>{{ credits.project }}</h1>
            <div class="row">
              <span class="badge badge-accent">v{{ credits.version }}</span>
              <span class="badge">{{ credits.license }} license</span>
              <span class="badge"><Icon name="lock" />Runs locally</span>
            </div>
          </div>
        </div>
        <p class="muted text-md">{{ credits.summary }}</p>
        <div class="grid-2">
          <div class="stack-sm">
            <div class="field-label">Install</div>
            <CopyField text="pip install easy-ml-studio" label="Copy install command" />
          </div>
          <div class="stack-sm">
            <div class="field-label">Start the studio</div>
            <CopyField text="easy-ml" label="Copy start command" />
          </div>
        </div>
        <div class="link-row">
          <a v-for="link in credits.links" :key="link.id" class="btn btn-secondary" :href="link.url" target="_blank" rel="noopener noreferrer">
            <Icon :name="LINK_ICONS[link.id] || 'open_in_new'" />{{ link.label }}
          </a>
          <a class="btn btn-ghost" :href="'mailto:' + credits.contact_email"><Icon name="mail_outline" />Contact the team</a>
        </div>
      </section>

      <section class="stack" v-if="credits">
        <div class="section-title"><h2>The team</h2><p>Easy-ML Studio is built and maintained by four developers.</p></div>
        <div class="grid-4">
          <div v-for="dev in credits.developers" :key="dev.name" class="team-card">
            <div class="avatar">{{ initials(dev.name) }}</div>
            <div class="stack-xs">
              <div class="team-name">{{ dev.name }}</div>
              <div class="subtle text-sm">{{ dev.role }}</div>
              <a v-if="dev.linkedin" class="text-sm" :href="dev.linkedin" target="_blank" rel="noopener noreferrer">LinkedIn</a>
            </div>
          </div>
        </div>
      </section>

      <section class="grid-2" v-if="credits">
        <div class="card">
          <div class="card-header"><h3>Contributing</h3></div>
          <div class="card-body muted text-sm">
            <p v-if="credits.show_repo_links">Bug reports, ideas and pull requests are welcome on GitHub. Use the Report a bug link above to open an issue.</p>
            <p v-else>The source repository will open to the public soon. Until then, reach the team through the contact link above.</p>
          </div>
        </div>
        <div class="card">
          <div class="card-header"><h3>Open-source software used</h3></div>
          <div class="card-body">
            <dl class="kv"><template v-for="item in credits.third_party" :key="item.name"><dt>{{ item.name }}</dt><dd>{{ item.license }}</dd></template></dl>
          </div>
        </div>
      </section>
    </div>`,
};
