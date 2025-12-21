/* assets/js/map_click_rail.js
 * Map is the navigator.
 * Clicking an overlay opens a full-height detail rail on the right.
 *
 * PATCH: guard optional fields (sections) + safe init with api.whenReady
 * PATCH 2025-12-21: support chord diagram HTML (iframe) + click-to-enlarge lightbox
 */

(function () {
  "use strict";

  const mapEl = document.getElementById("scrolly-map");
  if (!mapEl || !window.ADAEventMap) return;

  const prefersReducedMotion =
    window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;

  const storyRail = document.getElementById("story-rail");
  const eventRail = document.getElementById("event-rail");

  const btnHome = document.getElementById("btn-home");
  const btnReset = document.getElementById("btn-reset");
  const btnBack = document.getElementById("event-back");

  const elTitle = document.getElementById("event-title");
  const elDesc = document.getElementById("event-desc");
  const elCategory = document.getElementById("event-category");
  const elDate = document.getElementById("event-date");
  const elStats = document.getElementById("event-stats");
  const elMedia = document.getElementById("event-media");
  const elSections = document.getElementById("event-sections");

  // -----------------------
  // Helpers
  // -----------------------
  function escapeHtml(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function slugifyForChord(title) {
    return String(title || "")
      .toLowerCase()
      .replaceAll("&", "and")
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .replace(/_+/g, "_");
  }

  function showStory() {
    if (eventRail) eventRail.hidden = true;
    if (storyRail) storyRail.hidden = false;

    api.setCategoryFilter?.(null);
    renderCategoryLegend(null);
  }

  function showEvent() {
    if (storyRail) storyRail.hidden = true;
    if (eventRail) eventRail.hidden = false;
  }

  function clearEventRail() {
    if (elTitle) elTitle.textContent = "";
    if (elDesc) elDesc.textContent = "";
    if (elCategory) elCategory.textContent = "";
    if (elDate) elDate.textContent = "";
    if (elStats) elStats.innerHTML = "";
    if (elMedia) elMedia.innerHTML = "";
    if (elSections) elSections.innerHTML = "";
  }

  function statRow(label, value) {
    return `
      <div class="stat-row">
        <span class="stat-label">${escapeHtml(label)}</span>
        <span class="stat-value">${escapeHtml(value)}</span>
      </div>
    `;
  }

  // Media card supports:
  // - images: { type:"img", src:"...", caption:"..." }
  // - html chord diagrams: { type:"iframe", src:"...html", caption:"..." }
  function mediaCard(media) {
    const src = media?.src;
    if (!src) return "";

    const caption = media.caption || "";
    const type =
      media.type ||
      (String(src).toLowerCase().endsWith(".html") ? "iframe" : "img");

    const safeCaption = caption || "Event figure";

    if (type === "iframe") {
      // Preview iframe: keep it non-interactive so scrolling the rail stays smooth.
      // Clicking opens the lightbox where it becomes interactive.
      return `
        <div class="media-card" role="button" tabindex="0"
             data-media-type="iframe" data-media-src="${escapeHtml(src)}" data-media-caption="${escapeHtml(safeCaption)}">
          <div class="media-embed">
            <iframe
              src="${escapeHtml(src)}"
              title="${escapeHtml(safeCaption)}"
              loading="lazy"
              sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
            ></iframe>
          </div>
          ${caption ? `<div class="media-caption">${escapeHtml(caption)}</div>` : ""}
          <div class="media-zoom-hint">Click to enlarge</div>
        </div>
      `;
    }

    // Default: image
    return `
      <div class="media-card" role="button" tabindex="0"
           data-media-type="img" data-media-src="${escapeHtml(src)}" data-media-caption="${escapeHtml(safeCaption)}">
        <img src="${escapeHtml(src)}" alt="${escapeHtml(safeCaption)}" loading="lazy">
        ${caption ? `<div class="media-caption">${escapeHtml(caption)}</div>` : ""}
        <div class="media-zoom-hint">Click to enlarge</div>
      </div>
    `;
  }

  function sectionBlock(title, bodyHtml) {
    return `
      <div class="event-section">
        <h4>${escapeHtml(title)}</h4>
        ${bodyHtml}
      </div>
    `;
  }

  // -----------------------
  // Lightbox (modal)
  // -----------------------
  const lightbox = (() => {
    let root, bodyEl, captionEl, closeBtn;

    function ensure() {
      if (root) return;

      root = document.createElement("div");
      root.className = "media-lightbox";
      root.hidden = true;
      root.innerHTML = `
        <div class="media-lightbox-backdrop" data-close="1"></div>
        <div class="media-lightbox-dialog" role="dialog" aria-modal="true" aria-label="Media viewer">
          <button class="media-lightbox-close" type="button" aria-label="Close" data-close="1">×</button>
          <div class="media-lightbox-body"></div>
          <div class="media-lightbox-caption"></div>
        </div>
      `;
      document.body.appendChild(root);

      bodyEl = root.querySelector(".media-lightbox-body");
      captionEl = root.querySelector(".media-lightbox-caption");
      closeBtn = root.querySelector(".media-lightbox-close");

      root.addEventListener("click", (e) => {
        const t = e.target;
        if (t?.dataset?.close === "1") close();
      });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && root && !root.hidden) close();
      });
    }

    function open({ type, src, caption }) {
      if (!src) return;
      ensure();

      bodyEl.innerHTML = "";
      captionEl.textContent = caption || "";

      if (type === "iframe") {
        const iframe = document.createElement("iframe");
        iframe.src = src;
        iframe.title = caption || "Embedded figure";
        iframe.loading = "eager";
        iframe.className = "media-lightbox-iframe";
        iframe.setAttribute(
          "sandbox",
          "allow-scripts allow-same-origin allow-popups allow-forms"
        );
        bodyEl.appendChild(iframe);
      } else {
        const img = document.createElement("img");
        img.src = src;
        img.alt = caption || "Figure";
        img.className = "media-lightbox-img";
        bodyEl.appendChild(img);
      }

      root.hidden = false;
      document.documentElement.classList.add("has-media-lightbox");
      closeBtn?.focus?.();
    }

    function close() {
      if (!root) return;
      root.hidden = true;
      bodyEl.innerHTML = "";
      document.documentElement.classList.remove("has-media-lightbox");
    }

    return { open, close };
  })();

  window.ADA_MEDIA_LIGHTBOX = lightbox;


  // -----------------------
  // Render event rail
  // -----------------------
  function renderEventRail(ev, colors) {
    if (!ev) return;

    if (elTitle) elTitle.textContent = ev.title || "Event";
    if (elDesc) elDesc.textContent = ev.desc || "";

    if (elCategory) {
      elCategory.textContent = ev.category || "Event";
      const c =
        colors && ev.category && colors[ev.category] ? colors[ev.category] : "#111827";
      elCategory.style.borderColor = "rgba(0,0,0,0.10)";
      elCategory.style.background = "rgba(0,0,0,0.03)";
      elCategory.style.boxShadow = `0 0 0 6px ${c}22`;
    }

    if (elDate) elDate.textContent = ev.date || "";

    // Stats
    let statsHtml = "";
    if (ev.stats && typeof ev.stats === "object") {
      for (const [k, v] of Object.entries(ev.stats)) {
        statsHtml += statRow(k, v);
      }
    }
    if (elStats) elStats.innerHTML = statsHtml;

    // Media (explicit + auto chord fallback)
    let mediaList = Array.isArray(ev.media) ? [...ev.media] : [];

    // Auto-try a chord diagram html if none specified:
    // assets/notebook/img/chord_diagrams/chord_<slug>.html
    const hasChord = mediaList.some((m) =>
      String(m?.src || "").toLowerCase().includes("chord_")
    );

    if (!hasChord && ev?.title) {
      const slug = slugifyForChord(ev.title);
      mediaList.unshift({
        type: "iframe",
        src: `assets/notebook/img/chord_diagrams/chord_${slug}.html`,
        caption: "Chord diagram (cross-community linking)",
      });
    }

    let mediaHtml = "";
    for (const m of mediaList) {
      if (!m?.src) continue;
      mediaHtml += mediaCard(m);
    }
    if (elMedia) elMedia.innerHTML = mediaHtml;

    // Sections (guard)
    let sectionsHtml = "";
    if (Array.isArray(ev.sections)) {
      for (const s of ev.sections) {
        if (!s?.title || !s?.html) continue;
        sectionsHtml += sectionBlock(s.title, s.html);
      }
    }
    if (elSections) elSections.innerHTML = sectionsHtml;

    api.setCategoryFilter?.(ev.category || null);
    renderCategoryLegend(ev.category || null);

    showEvent();

    if (!prefersReducedMotion && eventRail) {
      eventRail.scrollTo?.({ top: 0, behavior: "smooth" });
    } else if (eventRail) {
      eventRail.scrollTo?.(0, 0);
    }
  }

  // Click-to-enlarge (event delegation)
  if (elMedia) {
    elMedia.addEventListener("click", (e) => {
      const card = e.target?.closest?.(".media-card");
      if (!card) return;

      const type = card.dataset.mediaType || "img";
      const src = card.dataset.mediaSrc;
      const caption = card.dataset.mediaCaption || "";

      if (!src) return;
      lightbox.open({ type, src, caption });
    });

    // Keyboard accessibility (Enter / Space)
    elMedia.addEventListener("keydown", (e) => {
      const card = e.target?.closest?.(".media-card");
      if (!card) return;
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      card.click();
    });
  }

  // -----------------------
  // Init embedded map
  // -----------------------
  const api = window.ADAEventMap.initEventMap({
    containerId: "scrolly-map",
    mode: "embedded",
    initialView: { center: [25, 10], zoom: 2 },
    // IMPORTANT: event_cards path default is assets/data/event_cards.json
  });

  // ----- Auto-generate narrative legend from api.colors -----
  const legendList = document.getElementById("category-legend-list");

  function categoryLabel(cat) {
    return cat;
  }

  function renderCategoryLegend(activeCategory) {
    if (!legendList) return;

    const entries = Object.entries(api.colors || {});
    legendList.innerHTML = entries
      .map(([cat, color]) => {
        const isActive = !!activeCategory && cat === activeCategory;
        const isFaded = !!activeCategory && cat !== activeCategory;

        return `
        <div class="legend-item ${isActive ? "is-active" : ""} ${
          isFaded ? "is-faded" : ""
        }">
          <span class="legend-swatch" style="background:${color};"></span>
          <span>${escapeHtml(categoryLabel(cat))}</span>
        </div>
      `;
      })
      .join("");
  }

  // Monkey-patch openEvent to also render the rail.
  const originalOpenEvent = api.openEvent;
  api.openEvent = function (id) {
    const ev = api.getEventById(id);
    if (ev) renderEventRail(ev, api.colors);
    try {
      originalOpenEvent.call(api, id);
    } catch (e) {}
  };

  // Buttons
  if (btnHome) btnHome.addEventListener("click", () => showStory());
  if (btnBack) btnBack.addEventListener("click", () => showStory());

  if (btnReset)
    btnReset.addEventListener("click", () => {
      api.reset();
      showStory();
    });

  // Default state
  showStory();
  clearEventRail();

  // Wait until events loaded, then render legend
  Promise.resolve(api.whenReady).then(() => renderCategoryLegend(null));

  // Expose click hook (called by map_core.js on overlay click)
  window.ADA_ON_EVENT_CLICK = (eventId) => {
    if (!eventId) return;
    api.focusEvent(eventId);
    api.openEvent(eventId);
  };
})();
