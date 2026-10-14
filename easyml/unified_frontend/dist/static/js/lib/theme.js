// Light / dark theme. The <head> bootstrap script applies the theme before first paint;
// this module keeps it in sync afterwards.
import { ref } from './vue.js';

const STORAGE_KEY = 'easyml.theme';
const media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

function readPreference() {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value === 'light' || value === 'dark' ? value : 'system';
  } catch (e) {
    return 'system';
  }
}

export const themePreference = ref(readPreference());
export const resolvedTheme = ref(resolve(themePreference.value));

function resolve(preference) {
  if (preference === 'light' || preference === 'dark') return preference;
  return media && media.matches ? 'dark' : 'light';
}

function apply() {
  resolvedTheme.value = resolve(themePreference.value);
  document.documentElement.setAttribute('data-theme', resolvedTheme.value);
}

export function setThemePreference(preference) {
  themePreference.value = preference;
  try {
    if (preference === 'system') localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, preference);
  } catch (e) { /* storage unavailable: keep the choice for this session only */ }
  apply();
}

export function toggleTheme() {
  setThemePreference(resolvedTheme.value === 'dark' ? 'light' : 'dark');
}

if (media) {
  const onChange = () => { if (themePreference.value === 'system') apply(); };
  if (media.addEventListener) media.addEventListener('change', onChange);
  else if (media.addListener) media.addListener(onChange);
}

apply();

// Read a CSS custom property (used to theme Chart.js).
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
