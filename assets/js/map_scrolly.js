/* assets/js/map_scrolly.js
 * Premium++ Scrollytelling:
 * - snap points inside steps container
 * - dynamic legend per active event
 * - timeline scrubber (range + dots), synced both ways
 */

(function () {
  "use strict";

  const mapEl = document.getElementById("scrolly-map");
  const stepsContainer = document.getElementById("scrolly-steps");
  if (!mapEl || !stepsContainer || !window.ADAEventMap) return;

  const panel = document.getElementById("analysis-panel");
  const titleEl = document.getElementById("panel-title");
  const dateEl = document.getElementById("panel-date");
  const contentEl = document.getElementById("panel-content");
  const statsEl = document.getElementById("panel-stats");
  const closeBtn = document.getElementById("panel-close");

  const btnReset = document.getElementById("btn-reset");
  const progressBar = document.getElementById("scrolly-progress-bar");

  // Legend
  const legendSwatch = document.getElementById("legend-swatch");
  const legendTitle = document.getElementById("legend-title");
  const legendSub = document.getElementById("legend-sub");
  const legendMeta = document.getElementById("legend-meta");

  // Timeline scrubber
  const timelineRange = document.getElementById("timeline-range");
  const timelineDots = document.getElementById("timeline-dots");
  const timelineLabel = document.getElementById("timeline-label");

  const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;

  // Initialize shared map in embedded mode
  const api = window.ADAEventMap.initEventMap({
    containerId: "scrolly-map",
    panel,
    titleEl,
    dateEl,
    contentEl,
    statsEl,
    closeBtn,
    mode: "embedded",
    initialView: { center: [25, 10], zoom: 2 },
  });

  if (btnReset) btnReset.addEventListener("click", () => {
    api.reset();
    // also scroll to first step
    const steps = Array.from(document.querySelectorAll(".step"));
    if (steps[0]) steps[0].scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
  });

  const steps = Array.from(document.querySelectorAll(".step"));
  if (!steps.length) return;

  // ----- Timeline build -----
  function prettyTitle(ev) {
    return (ev && (ev.title || ev.id)) ? (ev.title || ev.id) : "Event";
  }

  function buildTimeline() {
    if (!timelineRange || !timelineDots) return;

    timelineRange.min = "0";
    timelineRange.max = String(Math.max(0, steps.length - 1));
    timelineRange.step = "1";
    timelineRange.value = "0";

    // Dots
    timelineDots.innerHTML = "";
    steps.forEach((stepEl, idx) => {
      const id = stepEl.dataset.event;
      const ev = api.getEventById(id);

      const dot = document.createElement("div");
      dot.className = "timeline-dot";
      dot.dataset.idx = String(idx);

      const label = document.createElement("div");
      label.className = "dot-label";
      label.textContent = prettyTitle(ev);

      dot.appendChild(label);
      dot.addEventListener("click", () => goToIndex(idx, true));

      timelineDots.appendChild(dot);
    });
  }

  // ----- State helpers -----
  let activeIdx = 0;
  let programmatic = false;

  function setActiveStep(stepEl) {
    steps.forEach(s => s.classList.remove("is-active"));
    stepEl.classList.add("is-active");
  }

  function updateLegend(ev) {
    if (!legendSwatch || !legendTitle || !legendSub || !legendMeta) return;

    const color = api.colors?.[ev.category] || "#111827";
    legendSwatch.style.background = color;

    legendTitle.textContent = ev.title || "Event";
    legendSub.textContent = `${ev.date || ""} • ${ev.category || ""}`.trim();

    // Meta pills: show 2–3 “most useful” stats if present
    const pills = [];
    if (ev.stats?.["Propagation Score"]) pills.push(`Propagation: ${ev.stats["Propagation Score"]}`);
    if (ev.stats?.["Dominant Sentiment"]) pills.push(`Sentiment: ${ev.stats["Dominant Sentiment"]}`);
    if (ev.category) pills.push(`Type: ${ev.category}`);

    legendMeta.innerHTML = pills
      .slice(0, 3)
      .map(t => `<span class="legend-pill">${t}</span>`)
      .join("");
  }

  function updateTimelineUI(idx) {
    if (timelineRange) timelineRange.value = String(idx);

    if (timelineDots) {
      Array.from(timelineDots.children).forEach((el, i) => {
        if (!(el instanceof HTMLElement)) return;
        el.classList.toggle("is-active", i === idx);
      });
    }

    if (timelineLabel) {
      const stepEl = steps[idx];
      const ev = api.getEventById(stepEl?.dataset?.event);
      timelineLabel.textContent = ev ? `Timeline — ${ev.date || ev.title || "Event"}` : "Timeline";
    }
  }

  function activateIndex(idx, scrollStep) {
    idx = Math.max(0, Math.min(steps.length - 1, idx));
    if (idx === activeIdx && !scrollStep) return;

    activeIdx = idx;

    const stepEl = steps[idx];
    const id = stepEl.dataset.event;
    const ev = api.getEventById(id);
    if (!ev) return;

    if (scrollStep) {
      programmatic = true;
      stepEl.scrollIntoView({
        behavior: prefersReducedMotion ? "auto" : "smooth",
        block: "start",
        inline: "nearest",
      });
      setTimeout(() => { programmatic = false; }, prefersReducedMotion ? 0 : 250);
    }

    setActiveStep(stepEl);

    // Drive map + panel
    api.focusEvent(id);
    api.openEvent(id);

    // Drive legend + timeline UI
    updateLegend(ev);
    updateTimelineUI(idx);

    if (!prefersReducedMotion) api.resyncMap();
  }

  function goToIndex(idx, scrollStep) {
    activateIndex(idx, scrollStep);
  }

  // ----- IntersectionObserver within stepsContainer (snap flow) -----
  const observer = new IntersectionObserver((entries) => {
    if (programmatic) return;

    // Choose the most visible step within container
    const visible = entries
      .filter(e => e.isIntersecting)
      .sort((a, b) => (b.intersectionRatio || 0) - (a.intersectionRatio || 0))[0];

    if (!visible) return;

    const stepEl = visible.target;
    const idx = steps.indexOf(stepEl);
    if (idx >= 0) activateIndex(idx, false);

  }, {
    root: stepsContainer,
    threshold: [0.25, 0.45, 0.65, 0.85],
    rootMargin: "0px 0px -20% 0px"
  });

  steps.forEach(s => observer.observe(s));

  // ----- Progress bar based on steps container scroll -----
  function updateProgress() {
    if (!progressBar) return;

    const scrollTop = stepsContainer.scrollTop;
    const scrollH = stepsContainer.scrollHeight;
    const clientH = stepsContainer.clientHeight;
    const denom = Math.max(1, (scrollH - clientH));
    const ratio = Math.max(0, Math.min(1, scrollTop / denom));
    progressBar.style.width = (ratio * 100).toFixed(1) + "%";
  }

  stepsContainer.addEventListener("scroll", () => {
    updateProgress();
  }, { passive: true });

  // ----- Timeline interactions -----
  buildTimeline();

  if (timelineRange) {
    timelineRange.addEventListener("input", (e) => {
      const v = Number(e.target.value || 0);
      goToIndex(v, true);
    });
  }

  // ----- Default: activate first step -----
  goToIndex(0, false);
  updateProgress();

  // Resync on resize
  window.addEventListener("resize", () => {
    api.resyncMap();
    updateProgress();
  });
})();
