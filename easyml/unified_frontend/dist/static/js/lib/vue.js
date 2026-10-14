// Vue, Chart.js and Socket.IO are loaded as classic scripts from /static/vendor (no build step).
// Import them from here so every module uses the same instance.
const Vue = window.Vue;

export const {
  createApp, ref, reactive, computed, watch, nextTick,
  onMounted, onBeforeUnmount, onUnmounted, toRaw,
} = Vue;

export const Chart = window.Chart;
export const io = window.io;
