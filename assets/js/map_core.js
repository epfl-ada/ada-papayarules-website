/* assets/js/map_core.js
 * Shared Leaflet map + event overlays + panel rendering
 * Used by:
 *  - index.md (scrollytelling)
 *  - map.html (standalone explorer)
 */

(function () {
  "use strict";

  // Color scheme (kept from your original map)
  const colors = {
    "Political/Governance": "#1f77b4",
    "Conflict/Security": "#ff7f0e",
    "Disasters": "#2ca02c",
    "Diplomacy": "#d62728",
    "Climate/Environment": "#9467bd",
    "Health/Public Health": "#8c564b",
  };

  // Fallback palette for any new categories loaded from JSON
  const fallbackPalette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
  ];

  function ensureCategoryColor(category) {
    if (!category) return "#111827";
    if (colors[category]) return colors[category];
    const used = Object.keys(colors).length;
    colors[category] = fallbackPalette[used % fallbackPalette.length];
    return colors[category];
  }

  function slugify(s) {
    return String(s || "")
      .toLowerCase()
      .replace(/&/g, "and")
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  function titleKey(s) {
    const stop = new Set(["of","the","and","a","an","to","in","on","for","with","&"]);
    const toks = String(s || "")
      .toLowerCase()
      .replace(/&/g, "and")
      .replace(/[^a-z0-9]+/g, " ")
      .trim()
      .split(/\s+/)
      .filter(t => t && !stop.has(t));
    toks.sort();
    return toks.join("_");
  }

  function statsFromNotebookCard(card) {
    const out = {};
    if (card && Object.prototype.hasOwnProperty.call(card, "Propagation Score")) {
      const v = card["Propagation Score"];
      out["Propagation Score"] = (typeof v === "number") ? v.toFixed(2) : String(v);
    }
    if (card && Object.prototype.hasOwnProperty.call(card, "Dominant Sentiment")) {
      out["Dominant Sentiment"] = String(card["Dominant Sentiment"]);
    }
    if (card && Object.prototype.hasOwnProperty.call(card, "Key Subreddits")) {
      const ks = card["Key Subreddits"];
      if (Array.isArray(ks)) {
        out["Key Subreddits"] = ks.map(s => `r/${String(s).replace(/^r\//, "")}`).join(", ");
      } else if (typeof ks === "string") {
        const parts = ks.split(",").map(x => x.trim()).filter(Boolean);
        out["Key Subreddits"] = parts.map(s => s.startsWith("r/") ? s : `r/${s}`).join(", ");
      } else {
        out["Key Subreddits"] = "—";
      }
    }
    return out;
  }


  // Normalize incoming JSON card into the website event schema (safe defaults)
  function normalizeEventCard(raw, idx) {
    const ev = { ...(raw || {}) };

    if (!ev.id) ev.id = slugify(ev.title || `event_${idx}`);
    // Notebook export may use key `event` instead of `title`
    ev.title = ev.title || ev.event || "Event";
    ev.category = ev.category || "Event";
    ev.date = ev.date || "";
    ev.desc = ev.desc || "";

    if (!ev.stats || typeof ev.stats !== "object") ev.stats = {};

    const mapped = statsFromNotebookCard(ev);
    if (Object.keys(mapped).length) {
      ev.stats = { ...(ev.stats || {}), ...mapped };
    }

    if (!Array.isArray(ev.sections)) ev.sections = [];
    if (!Array.isArray(ev.media)) ev.media = [];

    // Keep optional geo fields if present (but notebook cards usually won't have them)
    if (!Array.isArray(ev.country_codes) && Array.isArray(ev.countryCodes)) ev.country_codes = ev.countryCodes;
    if (!Array.isArray(ev.geometries)) ev.geometries = [];

    ensureCategoryColor(ev.category);
    return ev;
  }

  async function loadEventsFromJSON(url) {
    const res = await fetch(url, { cache: "no-cache" });
    if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
    const data = await res.json();
    const arr = Array.isArray(data) ? data : (Array.isArray(data.events) ? data.events : null);
    if (!arr) throw new Error("Invalid JSON schema: expected array or {events:[...]}");
    return arr.map(normalizeEventCard);
  }

  // Merge notebook card metrics into existing overlay events (keeps geometries/country_codes)

  function mergeCardsIntoOverlayEvents(overlayEvents, loadedCards) {
    const byKey = new Map();

    loadedCards.forEach(c => {
      const srcTitle = c.title || c.event;
      const id = c.id ? String(c.id) : "";
      const kSlug = slugify(srcTitle);
      const kFuzzy = titleKey(srcTitle);

      if (id) byKey.set(id, c);
      if (kSlug) byKey.set(kSlug, c);
      if (kFuzzy) byKey.set(kFuzzy, c);
      if (srcTitle) byKey.set(String(srcTitle), c);
    });

    return overlayEvents.map(ev => {
      const candidates = [
        ev.id,
        slugify(ev.title),
        titleKey(ev.title),
        ev.title
      ].filter(Boolean);

      let card = null;
      for (const k of candidates) {
        if (byKey.has(k)) { card = byKey.get(k); break; }
      }
      if (!card) return ev;

      const cardTitle = card.title || card.event;

      return {
        ...ev,
        title: cardTitle || ev.title,
        category: card.category || ev.category,
        date: card.date || ev.date,
        desc: card.desc || ev.desc,
        stats: {
          ...(ev.stats || {}),
          ...statsFromNotebookCard(card),
        },
        sections: Array.isArray(card.sections) ? card.sections : ev.sections,
        media: Array.isArray(card.media) ? card.media : ev.media,
      };
    });
  }

  // Event data (kept from your original map; you can extend)
  let events = [
    {
      id: "crimea",
      title: "Crimea Annexation",
      category: "Political/Governance",
      date: "March 2014",
      geometries: [
        { type: "polygon", coords: [[46.2, 33.5], [46.0, 34.8], [45.3, 36.5], [44.4, 35.8], [44.4, 33.8], [45.5, 32.5]] }
      ],
      desc: "The annexation of Crimea by the Russian Federation sparked intense geopolitical debate. On Reddit, this event created a strong but isolated cluster of discussion within political subreddits.",
      stats: {
        "Propagation Score": "Low",
        "Dominant Sentiment": "Negative",
        "Key Subreddits": "r/UkrainianConflict, r/Russia, r/WorldNews"
      },
      media: [
        { src: "assets/img/event_propagation.png", caption: "Propagation compared across events." },
        { src: "assets/img/temporal_event_comparison.png", caption: "Temporal attention patterns." }
      ],
      sections: [
        { title: "What we observed", html: "<p>More detailed narrative…</p>" },
        { title: "Where it spread", html: "<ul><li>r/WorldNews</li><li>r/Politics</li></ul>" }
      ]
    },
    {
      id: "mh370",
      title: "MH370 Disappearance",
      category: "Disasters",
      date: "March 2014",
      geometries: [
        { type: "point", coords: [-38, 88] },
        { type: "point", coords: [3.13, 101.68] },
        { type: "point", coords: [39.9, 116.4] }
      ],
      desc: "The disappearance of Malaysia Airlines Flight 370 triggered a massive global mystery. Unlike political events, this generated cross-community speculation, linking aviation enthusiasts with conspiracy theorists.",
      stats: {
        "Propagation Score": "Medium",
        "Dominant Sentiment": "Neutral/Anxious",
        "Key Subreddits": "r/Aviation, r/Conspiracy, r/News"
      }
    },
    {
      id: "isis",
      title: "Rise of ISIS",
      category: "Conflict/Security",
      date: "June 2014",
      country_codes: ["SYR", "IRQ"],
      geometries: [
        { type: "point", coords: [48.85, 2.35] },
        { type: "point", coords: [50.85, 4.35] },
        { type: "point", coords: [51.50, -0.12] },
        { type: "point", coords: [28.53, -81.37] }
      ],
      desc: "The rapid expansion of ISIS in 2014 led to a surge in conflict footage and political debate. This event showed high modularity, with discussions largely contained within war-focused communities.",
      stats: {
        "Propagation Score": "Low",
        "Dominant Sentiment": "Very Negative",
        "Key Subreddits": "r/SyrianCivilWar, r/CombatFootage"
      }
    },
    {
      id: "ebola",
      title: "Ebola Outbreak",
      category: "Health/Public Health",
      date: "Late 2014",
      country_codes: ["GIN", "SLE", "LBR"],
      geometries: [
        { type: "point", coords: [32.77, -96.79] },
        { type: "point", coords: [40.41, -3.7] }
      ],
      desc: "The Ebola epidemic was a 'sudden shock' event. It created a brief but intense spike in cross-community links, driven by fear and health updates, before fading quickly.",
      stats: {
        "Propagation Score": "High (Short-term)",
        "Dominant Sentiment": "Fear/Concern",
        "Key Subreddits": "r/Ebola, r/Health, r/WorldNews"
      }
    },
    {
      id: "charlie",
      title: "Charlie Hebdo Attack",
      category: "Conflict/Security",
      date: "Jan 2015",
      geometries: [
        { type: "point", coords: [48.85, 2.35] }
      ],
      desc: "The attack on Charlie Hebdo triggered a 'Je Suis Charlie' solidarity wave. This event bridged political and non-political communities more effectively than typical political news.",
      stats: {
        "Propagation Score": "High",
        "Dominant Sentiment": "Solidarity/Anger",
        "Key Subreddits": "r/France, r/Europe, r/Pics"
      }
    },
    {
      id: "nepal",
      title: "Nepal Earthquake",
      category: "Disasters",
      date: "April 2015",
      country_codes: ["NPL"],
      desc: "A major natural disaster that triggered a 'solidarity' response. Unlike political events, links here were overwhelmingly positive, focused on donations and support.",
      stats: {
        "Propagation Score": "Medium",
        "Dominant Sentiment": "Positive (Supportive)",
        "Key Subreddits": "r/Nepal, r/Earthquakes, r/UpliftingNews"
      }
    },
    {
      id: "brexit",
      title: "Brexit Referendum",
      category: "Political/Governance",
      date: "June 2016",
      country_codes: ["GBR"],
      desc: "A classic 'slow burn' political event. Discussion built up over months. It was highly polarizing and stayed largely within political bubbles, with little spillover to general interest subs.",
      stats: {
        "Propagation Score": "Low",
        "Dominant Sentiment": "Polarized",
        "Key Subreddits": "r/UKPolitics, r/UnitedKingdom, r/Europe"
      }
    },
    {
      id: "us_election",
      title: "2016 US Election",
      category: "Political/Governance",
      date: "Nov 2016",
      country_codes: ["USA"],
      desc: "The most significant event in our dataset. Despite its global importance, it exhibited 'Structural Quarantine'. Political subs became hyper-active echo chambers, but the rest of Reddit largely ignored the daily drama.",
      stats: {
        "Propagation Score": "Very Low (Relative to size)",
        "Dominant Sentiment": "Highly Negative",
        "Key Subreddits": "r/The_Donald, r/Politics, r/SandersForPresident"
      }
    },
    {
      id: "nice",
      title: "Nice Truck Attack",
      category: "Conflict/Security",
      date: "July 2016",
      geometries: [
        { type: "point", coords: [43.71, 7.26] }
      ],
      desc: "Another 'shock' event. Similar to Charlie Hebdo, it caused a temporary spike in attention but didn't sustain long-term cross-community dialogue.",
      stats: {
        "Propagation Score": "Medium",
        "Dominant Sentiment": "Shock/Sadness",
        "Key Subreddits": "r/France, r/WorldNews"
      }
    },
    {
      id: "sk_scandal",
      title: "South Korean Political Scandal",
      category: "Political/Governance",
      date: "Late 2016",
      country_codes: ["KOR"],
      desc: "The impeachment of Park Geun-hye. While huge in Korea, on the English-speaking web it remained a niche topic, showing how language barriers reinforce network modularity.",
      stats: {
        "Propagation Score": "Very Low",
        "Dominant Sentiment": "Neutral",
        "Key Subreddits": "r/Korea, r/Geopolitics"
      }
    },
    {
      id: "el_chapo",
      title: "El Chapo Arrest",
      category: "Conflict/Security",
      date: "Feb 22, 2014",
      country_codes: ["MEX"],
      geometries: [
        { type: "point", coords: [25.79, -108.99] } // Los Mochis area (Sinaloa)
      ],
      desc: "Joaquín 'El Chapo' Guzmán’s arrest triggered a short burst of security-focused coverage. On Reddit, discussion clustered in news/crime communities with limited spillover to non-news spaces.",
      stats: {
        "Propagation Score": "Low–Medium",
        "Dominant Sentiment": "Neutral/Alarm",
        "Key Subreddits": "r/worldnews, r/news, r/TrueCrime"
      }
    },

    {
      id: "cyclone_pam",
      title: "Cyclone Pam (Vanuatu)",
      category: "Disasters",
      date: "Mar 13–14, 2015",
      country_codes: ["VUT"],
      geometries: [
        { type: "point", coords: [-17.73, 168.32] } // Port Vila
      ],
      desc: "A severe cyclone that generated a donation/relief pulse. Compared to political crises, disaster attention tends to bridge communities briefly via fundraising and on-the-ground updates.",
      stats: {
        "Propagation Score": "Medium (Short-term)",
        "Dominant Sentiment": "Concern/Support",
        "Key Subreddits": "r/worldnews, r/earthquakes, r/UpliftingNews"
      }
    },

    {
      id: "fort_mcmurray",
      title: "Fort McMurray Wildfire",
      category: "Disasters",
      date: "May 2016",
      country_codes: ["CAN"],
      geometries: [
        { type: "point", coords: [56.73, -111.38] } // Fort McMurray
      ],
      desc: "The wildfire produced sustained coverage in Canadian and general news communities, with practical information-sharing (evacuation, air quality) driving cross-links more than ideology.",
      stats: {
        "Propagation Score": "Medium",
        "Dominant Sentiment": "Concern/Sympathy",
        "Key Subreddits": "r/canada, r/worldnews, r/news"
      }
    },

    {
      id: "jcpao",
      title: "Iran Nuclear Deal (JCPOA) Signed",
      category: "Diplomacy",
      date: "Jul 14, 2015",
      country_codes: ["IRN", "USA"],
      geometries: [
        { type: "point", coords: [48.21, 16.37] } // Vienna (talks venue)
      ],
      desc: "The JCPOA announcement triggered a concentrated spike in geopolitics communities. Most bridging occurred between policy-focused subreddits rather than broad lifestyle/hobby spaces.",
      stats: {
        "Propagation Score": "Low–Medium",
        "Dominant Sentiment": "Polarized/Analytical",
        "Key Subreddits": "r/geopolitics, r/worldnews, r/politics"
      }
    },

    {
      id: "obama_cuba",
      title: "Obama Visits Cuba",
      category: "Diplomacy",
      date: "Mar 20–22, 2016",
      country_codes: ["CUB", "USA"],
      geometries: [
        { type: "point", coords: [23.11, -82.37] } // Havana
      ],
      desc: "A symbolic diplomatic moment with culturally-oriented coverage (history, photos) that briefly linked politics to general-interest communities.",
      stats: {
        "Propagation Score": "Medium (Brief)",
        "Dominant Sentiment": "Curiosity/Mixed",
        "Key Subreddits": "r/worldnews, r/pics, r/history"
      }
    },

    {
      id: "colombia_farc",
      title: "Colombia–FARC Peace Accord",
      category: "Diplomacy",
      date: "Sep 26, 2016",
      country_codes: ["COL"],
      geometries: [
        { type: "point", coords: [4.71, -74.07] } // Bogotá
      ],
      desc: "The accord generated relatively niche but constructive discussion; bridging mainly happened via regional subreddits and global-news hubs.",
      stats: {
        "Propagation Score": "Low–Medium",
        "Dominant Sentiment": "Hopeful/Analytical",
        "Key Subreddits": "r/worldnews, r/Colombia, r/geopolitics"
      }
    },

    {
      id: "turkish_coup",
      title: "Turkish Coup Attempt",
      category: "Conflict/Security",
      date: "Jul 15–16, 2016",
      country_codes: ["TUR"],
      geometries: [
        { type: "point", coords: [39.93, 32.86] }, // Ankara
        { type: "point", coords: [41.01, 28.98] }  // Istanbul
      ],
      desc: "A sudden security shock that produced intense real-time updates. Reddit bridging spiked briefly as users relayed live reporting, then quickly recentralized into news/politics clusters.",
      stats: {
        "Propagation Score": "Medium (Short-term)",
        "Dominant Sentiment": "Shock/Uncertainty",
        "Key Subreddits": "r/worldnews, r/Turkey, r/news"
      }
    },

    {
      id: "us_campaign",
      title: "US Presidential Campaign (2016)",
      category: "Political/Governance",
      date: "Jun 2015 – Nov 2016",
      country_codes: ["USA"],
      geometries: [
        { type: "point", coords: [38.90, -77.04] } // Washington, DC (symbolic)
      ],
      desc: "A long-running political arc. Linking activity was high inside political subreddits, but cross-community spillover remained limited—consistent with structural quarantine.",
      stats: {
        "Propagation Score": "Very Low (Outside politics)",
        "Dominant Sentiment": "Highly Negative/Polarized",
        "Key Subreddits": "r/politics, r/SandersForPresident, r/The_Donald"
      }
    },

    {
      id: "trump_inaug",
      title: "Donald Trump Inauguration",
      category: "Political/Governance",
      date: "Jan 20, 2017",
      country_codes: ["USA"],
      geometries: [
        { type: "point", coords: [38.89, -77.03] } // National Mall / DC
      ],
      desc: "A concentrated peak event that amplified existing political communities. Bridging was mostly intra-politics; general-interest subreddits saw only brief ‘headline’ attention.",
      stats: {
        "Propagation Score": "Low",
        "Dominant Sentiment": "Polarized",
        "Key Subreddits": "r/politics, r/The_Donald, r/worldnews"
      }
    },

    {
      id: "un_climate_nyc",
      title: "UN Climate Summit (NYC)",
      category: "Climate/Environment",
      date: "Sep 23, 2014",
      geometries: [
        { type: "point", coords: [40.71, -74.01] } // NYC
      ],
      desc: "High-level climate diplomacy tends to remain policy-centric. On Reddit, links largely stayed within science/policy spheres, with limited crossover to mainstream communities.",
      stats: {
        "Propagation Score": "Low",
        "Dominant Sentiment": "Analytical/Concern",
        "Key Subreddits": "r/environment, r/science, r/worldnews"
      }
    },

    {
      id: "cop21",
      title: "COP21 Climate Conference (Paris)",
      category: "Climate/Environment",
      date: "Nov 30 – Dec 12, 2015",
      geometries: [
        { type: "point", coords: [48.85, 2.35] } // Paris
      ],
      desc: "COP21 produced sustained but specialized attention. The strongest bridges formed between environment/science subreddits and general news hubs.",
      stats: {
        "Propagation Score": "Low–Medium",
        "Dominant Sentiment": "Concern/Pragmatic",
        "Key Subreddits": "r/science, r/environment, r/worldnews"
      }
    },

    {
      id: "cop22",
      title: "COP22 Marrakech",
      category: "Climate/Environment",
      date: "Nov 7–18, 2016",
      country_codes: ["MAR"],
      geometries: [
        { type: "point", coords: [31.63, -7.99] } // Marrakech
      ],
      desc: "Follow-up climate negotiations drew modest attention compared to COP21. Bridging remained limited and concentrated in science/environment clusters.",
      stats: {
        "Propagation Score": "Very Low",
        "Dominant Sentiment": "Concern/Low salience",
        "Key Subreddits": "r/environment, r/science, r/worldnews"
      }
    },

  ];

  function getEventById(id) {
    return events.find(e => e.id === id) || null;
  }

  // Panel rendering (works in index + standalone)
  function renderPanel(ev, panelEls) {
    const { panel, titleEl, dateEl, contentEl, statsEl } = panelEls;
    if (!panel || !titleEl || !dateEl || !contentEl || !statsEl) return;

    titleEl.innerText = ev.title;
    titleEl.style.color = colors[ev.category] || "#111827";

    dateEl.innerText = `${ev.date} • ${ev.category}`;
    contentEl.innerHTML = `<p>${ev.desc || ""}</p>`;

    let statsHtml = "";
    if (ev.stats) {
      for (const [key, value] of Object.entries(ev.stats)) {
        statsHtml += `
          <div class="stat-row">
            <span class="stat-label">${key}</span>
            <span class="stat-value">${value}</span>
          </div>
        `;
      }
    }
    statsEl.innerHTML = statsHtml;

    panel.classList.add("active");
  }

  function closePanel(panelEls) {
    if (panelEls?.panel) panelEls.panel.classList.remove("active");
  }

  // Helpers for wrapping overlays (prevents dateline weirdness on wide screens)
  function shiftLng(lng, k) { return lng + 360 * k; }

  function shiftGeoJSONGeometryCoords(coords, k) {
    if (!Array.isArray(coords)) return coords;
    if (coords.length === 2 && typeof coords[0] === "number" && typeof coords[1] === "number") {
      return [coords[0] + 360 * k, coords[1]];
    }
    return coords.map(c => shiftGeoJSONGeometryCoords(c, k));
  }

  function shiftedFeatures(features, k) {
    if (k === 0) return features;
    return features.map(f => {
      const g = f.geometry;
      return {
        ...f,
        geometry: {
          ...g,
          coordinates: shiftGeoJSONGeometryCoords(g.coordinates, k)
        }
      };
    });
  }

  function shiftLatLng(latlng, k) { return [latlng[0], shiftLng(latlng[1], k)]; }

  function shiftLatLngs(latlngs, k) {
    if (!Array.isArray(latlngs)) return latlngs;
    if (latlngs.length === 2 && typeof latlngs[0] === "number" && typeof latlngs[1] === "number") {
      return shiftLatLng(latlngs, k);
    }
    return latlngs.map(p => shiftLatLngs(p, k));
  }

  function makeNeededWrapKs(map) {
    const z = map.getZoom();
    const worldPx = 256 * Math.pow(2, z);
    const halfW = map.getSize().x / 2;
    let n = Math.ceil(halfW / worldPx) + 1;
    n = Math.min(n, 6);
    const ks = [];
    for (let k = -n; k <= n; k++) ks.push(k);
    return ks;
  }

  // Bounds helpers to focus event
  function computeEventBounds(L, map, ev, overlayRootLayerGroup) {
    // We compute bounds by building temporary LatLngs where possible.
    // If no geometry available, fallback to world view.
    const bounds = L.latLngBounds();

    let added = false;

    if (ev.country_codes && window.worldGeoJSON?.features?.length) {
      const feats = window.worldGeoJSON.features.filter(f => {
        const props = f.properties || {};
        const code = f.id || props.ISO_A3 || props.ADM0_A3 || props.SOV_A3;
        return code && ev.country_codes.includes(code);
      });

      // Convert GeoJSON bbox-ish by iterating coords
      for (const feat of feats) {
        const coords = feat?.geometry?.coordinates;
        if (!coords) continue;

        const walk = (c) => {
          if (!Array.isArray(c)) return;
          if (c.length === 2 && typeof c[0] === "number" && typeof c[1] === "number") {
            // GeoJSON is [lng, lat]
            bounds.extend([c[1], c[0]]);
            added = true;
            return;
          }
          for (const child of c) walk(child);
        };
        walk(coords);
      }
    }

    if (ev.geometries?.length) {
      for (const g of ev.geometries) {
        if (g.type === "point" && Array.isArray(g.coords) && g.coords.length === 2) {
          bounds.extend(g.coords);
          added = true;
        }
        if (g.type === "polygon" && Array.isArray(g.coords)) {
          for (const pt of g.coords) {
            if (Array.isArray(pt) && pt.length === 2) {
              bounds.extend(pt);
              added = true;
            }
          }
        }
      }
    }

    if (!added) {
      // fallback
      return null;
    }
    return bounds;
  }

  // Main initializer
  function initEventMap(opts) {
    const {
      containerId,
      panel,
      titleEl,
      dateEl,
      contentEl,
      statsEl,
      closeBtn,
      mode = "embedded", // "embedded" or "standalone"
      initialView = { center: [25, 10], zoom: 2 },
      eventsUrl = "assets/data/event_cards.json",
    } = opts;

    if (!window.L) throw new Error("Leaflet not found (window.L missing).");
    const L = window.L;

    const map = L.map(containerId, {
      zoomAnimation: false,
      fadeAnimation: false,
      markerZoomAnimation: false,
      zoomControl: false,
    }).setView(initialView.center, initialView.zoom);

    L.control.zoom({ position: "bottomleft" }).addTo(map);

    // Tiles
    L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(map);

    // Legend (standalone only by default; for embedded it can feel busy)
    if (mode === "standalone") {
      const legend = L.control({ position: "bottomright" });
      legend.onAdd = function () {
        const div = L.DomUtil.create("div", "legend");
        div.style.padding = "10px";
        div.style.background = "rgba(255,255,255,0.90)";
        div.style.boxShadow = "0 0 15px rgba(0,0,0,0.2)";
        div.style.borderRadius = "8px";
        div.style.lineHeight = "22px";
        div.style.color = "#555";
        for (const category in colors) {
          div.innerHTML += `<i style="display:inline-block;width:14px;height:14px;border-radius:4px;margin-right:8px;background:${colors[category]};opacity:0.75;"></i>${category}<br>`;
        }
        return div;
      };
      legend.addTo(map);
    }

    // Shared renderer improves performance for polygons
    const sharedSvgRenderer = L.svg({ padding: 0 });

    const overlayRoot = L.layerGroup().addTo(map);

    let activeCategoryFilter = null;

function applyCategoryFadeToLayer(layer, activeCategory) {
  const cat = layer.__adaCategory || null;
  const isMatch = !activeCategory || (cat === activeCategory);

  // If this is a LayerGroup/GeoJSON, apply to its children
  if (layer.eachLayer) {
    layer.eachLayer(child => applyCategoryFadeToLayer(child, activeCategory));
    return;
  }

  // Polygons / paths
  if (layer.setStyle) {
    layer.setStyle({
      opacity: isMatch ? 1 : 0.18,
      fillOpacity: isMatch ? 0.35 : 0.06,
    });
    return;
  }

  // Markers (divIcon)
  const el = layer.getElement?.();
  if (el) {
    el.style.opacity = isMatch ? "1" : "0.18";
    el.style.filter = isMatch ? "none" : "grayscale(0.9)";
  }
}

function applyCategoryFade(activeCategory) {
  overlayRoot.eachLayer(l => applyCategoryFadeToLayer(l, activeCategory));
  // keep the map sharp after style changes
  requestAnimationFrame(() => map.invalidateSize(true));
}

function setCategoryFilter(categoryOrNull) {
  activeCategoryFilter = categoryOrNull || null;
  applyCategoryFade(activeCategoryFilter);
}


    let currentWrapKsKey = "";

    function addInteraction(layer, ev, type, panelEls) {
      layer.on("click", (e) => {
        // NEW: route click into page-level UI (right rail) if available
        if (typeof window.ADA_ON_EVENT_CLICK === "function") {
          window.ADA_ON_EVENT_CLICK(ev.id);
        }

        renderPanel(ev, panelEls);
        L.DomEvent.stopPropagation(e);

        if (type === "point") {
          map.flyTo(e.latlng, Math.max(map.getZoom(), 5), { animate: false, duration: 1.0 });
        } else {
          const b = layer.getBounds?.();
          if (b && b.isValid()) {
            map.flyToBounds(b, { padding: [50, 50], animate: false, duration: 1.0 });
          }
        }
      });

      layer.on("mouseover", () => {
        if (type === "polygon" && layer.setStyle) layer.setStyle({ fillOpacity: 0.65 });
      });
      layer.on("mouseout", () => {
        if (type === "polygon" && layer.setStyle) layer.setStyle({ fillOpacity: 0.35 });
      });
    }

    function buildOverlaysForK(k, panelEls) {
      const group = L.layerGroup();

      events.forEach(ev => {
        const color = colors[ev.category] || "#111827";

        // Country polygons
        if (ev.country_codes && window.worldGeoJSON?.features?.length) {
          const features = window.worldGeoJSON.features.filter(f => {
            const props = f.properties || {};
            const code = f.id || props.ISO_A3 || props.ADM0_A3 || props.SOV_A3;
            return code && ev.country_codes.includes(code);
          });

          if (features.length > 0) {
            const layer = L.geoJSON(shiftedFeatures(features, k), {
              style: {
                color: color,
                fillColor: color,
                fillOpacity: 0.35,
                weight: 1,
              },
              smoothFactor: 1,
              renderer: sharedSvgRenderer,
            });
            layer.__adaCategory = ev.category;
            layer.addTo(group);
            addInteraction(layer, ev, "polygon", panelEls);
          }
        }

        // Manual geometries
        if (ev.geometries?.length) {
          ev.geometries.forEach(geo => {
            let layer = null;

            if (geo.type === "polygon") {
              const shifted = shiftLatLngs(geo.coords, k);
              layer = L.polygon(shifted, {
                color: color,
                fillColor: color,
                fillOpacity: 0.35,
                weight: 2,
                renderer: sharedSvgRenderer,
              });
              layer.__adaCategory = ev.category;
              layer.addTo(group);
              addInteraction(layer, ev, "polygon", panelEls);
            }

            if (geo.type === "point") {
              const icon = L.divIcon({
                className: "leaflet-div-icon",
                html: `<div style="
                  width: 16px; height: 16px;
                  border-radius: 999px;
                  background: ${color};
                  border: 2px solid white;
                  box-shadow: 0 0 0 0 rgba(0,0,0,0.55);
                  animation: pulse 2s infinite;
                "></div>`,
                iconSize: [16, 16],
                iconAnchor: [8, 8],
              });

              const shiftedPt = shiftLatLng(geo.coords, k);
              layer = L.marker(shiftedPt, { icon });
              layer.__adaCategory = ev.category;
              layer.addTo(group);
              addInteraction(layer, ev, "point", panelEls);
            }
          });
        }
      });

      return group;
    }

    function updateWrappedOverlays(force, panelEls) {
      const ks = makeNeededWrapKs(map);
      const key = ks.join(",");

      if (!force && key === currentWrapKsKey) return;
      currentWrapKsKey = key;

      overlayRoot.clearLayers();
      ks.forEach(k => overlayRoot.addLayer(buildOverlaysForK(k, panelEls)));

      // Re-apply fade after rebuild (zoom/resizes recreate layers)
      if (activeCategoryFilter) applyCategoryFade(activeCategoryFilter);

    }

    const panelEls = {
      panel: panel || document.getElementById("analysis-panel"),
      titleEl: titleEl || document.getElementById("panel-title"),
      dateEl: dateEl || document.getElementById("panel-date"),
      contentEl: contentEl || document.getElementById("panel-content"),
      statsEl: statsEl || document.getElementById("panel-stats"),
    };

    // Initial overlays
    updateWrappedOverlays(true, panelEls);
    map.on("zoomend", () => updateWrappedOverlays(false, panelEls));
    window.addEventListener("resize", () => updateWrappedOverlays(true, panelEls));


    // PATCH: Load notebook-derived event cards and merge their metrics into the
    // existing overlay events (we keep geometries/country_codes in this file).
    const whenReady = (async () => {
      try {
        const loaded = await loadEventsFromJSON(eventsUrl);
        if (loaded && loaded.length) {
          events = mergeCardsIntoOverlayEvents(events, loaded);
          // Ensure colors include any new categories from JSON
          events.forEach(ev => ensureCategoryColor(ev.category));
          updateWrappedOverlays(true, panelEls);
          if (activeCategoryFilter) applyCategoryFade(activeCategoryFilter);
        }
      } catch (e) {
        console.warn("[ADAEventMap] event_cards.json not loaded; using built-in event data.", e);
      }
      return true;
    })();


    // Close panel
    if (closeBtn) closeBtn.addEventListener("click", () => closePanel(panelEls));
    map.on("click", () => closePanel(panelEls));

    // Make sure map sizes correctly if container changes
    function resyncMap() {
      requestAnimationFrame(() => map.invalidateSize(true));
      setTimeout(() => map.invalidateSize(true), 120);
    }
    window.addEventListener("load", resyncMap);
    window.addEventListener("resize", resyncMap);

    function focusEvent(idOrEvent) {
      const ev = typeof idOrEvent === "string" ? getEventById(idOrEvent) : idOrEvent;
      if (!ev) return;

      const bounds = computeEventBounds(L, map, ev, overlayRoot);
      if (bounds && bounds.isValid && bounds.isValid()) {
        map.flyToBounds(bounds, { padding: [60, 60], animate: false, duration: 1.0 });
        return;
      }

      // Fallback to event points if any
      const pt = ev?.geometries?.find(g => g.type === "point")?.coords;
      if (pt) {
        map.flyTo(pt, Math.max(map.getZoom(), 5), { animate: false, duration: 1.0 });
        return;
      }

      // Final fallback
      map.flyTo(initialView.center, initialView.zoom, { animate: false, duration: 1.0 });
    }

    function openEvent(id) {
      const ev = getEventById(id);
      if (!ev) return;
      renderPanel(ev, panelEls);
    }

    function reset() {
      closePanel(panelEls);
      setCategoryFilter(null);
      map.flyTo(initialView.center, initialView.zoom, { animate: false, duration: 1.0 });
    }


    return {
      map,
      get events() { return events; },
      colors,
      getEventById,
      focusEvent,
      openEvent,
      closePanel: () => closePanel(panelEls),
      reset,
      resyncMap,
      setCategoryFilter,
      whenReady,
    };
  }

  // Expose to window
  window.ADAEventMap = {
    initEventMap,
    getEventById,
    get events() { return events; },
    colors,
  };

  // Simple pulse animation for point markers (injected once)
  const style = document.createElement("style");
  style.textContent = `
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(0,0,0,0.45); }
      70% { box-shadow: 0 0 0 12px rgba(0,0,0,0); }
      100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); }
    }
  `;
  document.head.appendChild(style);

})();
