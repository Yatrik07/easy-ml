// Chart.js helpers: theme colours from CSS tokens and a lifecycle that never leaves a detached canvas.
import { Chart, nextTick, onBeforeUnmount, onMounted, watch } from './vue.js';
import { cssVar, resolvedTheme } from './theme.js';

export function chartColors() {
  const series = [];
  for (let i = 1; i <= 8; i++) series.push(cssVar('--chart-' + i));
  return { text: cssVar('--text-muted'), grid: cssVar('--chart-grid'), ref: cssVar('--chart-ref'), series };
}

export function seriesColor(colors, index) {
  return colors.series[index % colors.series.length];
}

// Axis options shared by the result charts.
export function axis(colors, title, extra) {
  return Object.assign({
    title: { display: !!title, text: title, color: colors.text },
    ticks: { color: colors.text },
    grid: { color: colors.grid },
  }, extra || {});
}

// Build the chart after mount, rebuild when `source()` changes or the theme switches, destroy on unmount.
// `buildConfig(colors)` returns a Chart.js config, or null to draw nothing.
export function useChart(canvasRef, buildConfig, source) {
  let chart = null;

  function build() {
    if (chart) { chart.destroy(); chart = null; }
    if (!canvasRef.value || !Chart) return;
    const config = buildConfig(chartColors());
    if (config) chart = new Chart(canvasRef.value.getContext('2d'), config);
  }

  onMounted(() => nextTick(build));
  if (source) watch(source, () => nextTick(build));
  watch(resolvedTheme, () => nextTick(build));
  onBeforeUnmount(() => { if (chart) chart.destroy(); chart = null; });
}
