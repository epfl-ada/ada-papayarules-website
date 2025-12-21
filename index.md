---
layout: full
---

<!-- Leaflet -->
<link
  rel="stylesheet"
  href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
  integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
  crossorigin=""
/>
<script
  src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
  integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
  crossorigin=""
></script>

<!-- World polygons -->
<script src="assets/js/world_data.js"></script>


<div class="hero-section">
  <div class="hero-inner">
    <div class="hero-kicker">EPFL Applied Data Analysis - Fall 2025</div>
    <h1>The Reddit Event Network</h1>
    <h2>Does Information Really Flow Across Communities?</h2>

    <div class="hero-cta-row">
      <a class="button button-ghost" href="map.html">Explore the map</a>
      <a class="button button-ghost" href="https://github.com/epfl-ada/ada-2025-project-papayarules/blob/main/results.ipynb">Notebook</a>
    </div>

    <div class="hero-scroll-hint">
      <span class="hint-dot"></span>
      Click regions on the map to reveal the story
    </div>
  </div>
</div>

<div class="content-container">

<section class="intro">
  <h2>The Question</h2>
  <div class="split">
    <div class="intro-left">
      <p>Scroll through Reddit long enough and it feels inevitable: polarized debates are everywhere. One thread bleeds into the next, outrage spreads, and suddenly every community looks like part of the same argument.</p>

      <p>But platforms don’t work by feeling. They work by structure. Reddit isn’t one crowd, it’s thousands of semi-isolated communities, each with its own borders. So we asked a more uncomfortable question: <strong>when event attention appears, does it actually travel?</strong></p>

      <p>We took that question literally. We tracked <strong>3 years of Reddit hyperlinks</strong> (2014–2017). Every link from one subreddit to another is a deliberate bridge, an attempt to pull context, attention, and narrative across a boundary. If polarization really spreads, those bridges should be everywhere.</p>

      <div class="highlight-box">
        <strong>Spoiler: Barely.</strong> Reddit isn't a polarized warzone where sensitive content infects every corner. It's more like a <strong>quarantine ward</strong>.
      </div>
    </div>

    <div class="intro-right right-stack">
      <div class="card">
        <h3>What We Looked At</h3>
        <p><strong>Timeline:</strong> 2014–2017</p>
        <p><strong>Signal:</strong> Cross-subreddit hyperlinks</p>
        <p><strong>Goal:</strong> See if polarizing content actually jumps communities.</p>
      </div>

      <div class="card">
        <h3>What “Propagation” Means</h3>
        <p>We measure how often an event produces <strong>cross-community links</strong> (bridges) beyond its “home” subreddits.</p>
        <p class="muted">Next: use the map to open an event and inspect its bridges.</p>
      </div>
    </div>
  </div>
</section>

<hr>

<section class="data-overview">
  <h2>The Data</h2>

  <div class="stats-grid">
    <div class="stat-item">
      <span class="stat-number">858k</span>
      <span class="stat-label">Hyperlinks</span>
    </div>
    <div class="stat-item">
      <span class="stat-number">27k</span>
      <span class="stat-label">Subreddits</span>
    </div>
    <div class="stat-item">
      <span class="stat-number">18</span>
      <span class="stat-label">Global Events</span>
    </div>
  </div>

  <div class="card">
    <figure style="margin: 0;">
      <img src="assets/img/timeline.png" alt="Timeline of Events">
      <figcaption>Timeline of major global events analyzed.</figcaption>
    </figure>
  </div>

  <div class="card">
    <figure style="margin:0;">
      <img
        src="assets/img/monthly_cross_community_activity.png"
        alt="Monthly cross-community linking activity with major global events"
      >
      <figcaption>
        Monthly cross-community hyperlink volume (2014–2017).  
        Vertical markers indicate major global events.  
        Spikes align with shocks but most activity quickly recedes.
      </figcaption>
    </figure>
  </div>


</section>

<hr>

<section id="scrolly" class="scrolly">
  <!-- LEFT: sticky map area (now horizontal split: map top + media bottom) -->
  <div class="scrolly-sticky" aria-label="Interactive map">
    <div class="scrolly-topbar">
      <div class="scrolly-title">
        <span class="badge">Interactive</span>
        Global Event Map
      </div>

      <div class="scrolly-actions">
        <button class="pill" id="btn-reset" type="button" title="Reset map view">Reset</button>
        <a class="pill pill-link" href="map.html" title="Open standalone explorer">Explorer</a>
      </div>
    </div>

    <!-- PATCH: horizontal split -->
    <div class="sticky-split-h">
      <!-- Top 2/3: map -->
      <div class="sticky-map">
        <div id="scrolly-map"></div>
      </div>
    </div>
  </div>

  <!-- RIGHT: story OR event detail -->
  <div class="scrolly-right">

    <!-- Default: Story rail (visible initially) -->
    <div id="story-rail" class="story-rail" aria-label="Story rail">
      <div class="story-rail-header">
        <div>
          <div class="rail-kicker">Narrative</div>
          <h3 class="rail-title">What the map reveals</h3>
        </div>
      </div>

      <div class="rail-block card">
        <h4>The Core Finding</h4>
        <p>Event-related subreddits form tight clusters and rarely link out. Most of Reddit remains structurally insulated from debates.</p>
        <p class="muted">Now click an event region on the map to see the evidence and examples.</p>
      </div>

      <div class="rail-block card">
  <h4>Which events actually propagate?</h4>

  <div class="media-card"
       role="button"
       tabindex="0"
       onclick="window.ADA_MEDIA_LIGHTBOX.open({
         type: 'img',
         src: 'assets/img/rq1_event_activity.png',
         caption: 'Event community activity (RQ1)'
       })">
    <img src="assets/img/rq1_event_activity.png" alt="RQ1 event activity">
    <div class="media-zoom-hint">Click to enlarge</div>
  </div>
</div>




      <!-- Auto legend will be injected here -->
<div id="category-legend" class="rail-block card legend-card" aria-label="Event category legend">
  <h4>Event categories</h4>
  <div id="category-legend-list" class="legend-list"></div>
  <p class="muted" style="margin-top:10px;">
    Colors indicate the dominant event type.
  </p>
</div>


   
      <div class="rail-block">
        <article class="step passive">
          <div class="step-kicker">Try this</div>
          <h3>Click Crimea, then US Election</h3>
          <p>Notice how events stay trapped inside their related spaces.</p>
        </article>

        <article class="step passive">
          <div class="step-kicker">Then compare</div>
          <h3>Click Ebola or Nepal</h3>
          <p>Disasters create brief cross-community bridges driven by solidarity.</p>
        </article>

        <article class="step passive">
          <div class="step-kicker">Interpretation</div>
          <h3>Echo chamber ≠ total contamination</h3>
          <p>Reddit looks polarized if you focus on event-related subs, but the network structure shows quarantine from the mainstream.</p>
        </article>
      </div>

      <div class="rail-block card">
        <h4>What you can learn here</h4>
        <ul>
          <li>Which event types bridge communities</li>
          <li>Which events remain contained</li>
          <li>How geography relates to discussion clusters</li>
        </ul>
      </div>

      <div class="rail-block subtle">
        <p class="muted">Open an event to see the deep-dive card.</p>
      </div>
    </div>

    <!-- Event detail (hidden until user clicks an overlay) -->
    <aside id="event-rail" class="event-rail" aria-label="Event detail panel" hidden>
      <h2 class="event-title" id="event-title">Event title</h2>
      <p class="event-desc" id="event-desc">Event description...</p>

      <!-- Stats -->
      <div class="event-stats" id="event-stats"></div>

      <!-- Media / figures (optional per event) -->
      <div class="event-media" id="event-media"></div>

      <!-- Extra sections -->
      <div class="event-sections" id="event-sections"></div>

      <div class="event-footer">
        <a class="button" href="https://github.com/epfl-ada/ada-2025-project-papayarules/blob/main/results.ipynb">See methods in notebook</a>
      </div>
    </aside>

  </div>
</section>

<hr>

<section class="conclusion">
  <h2>Conclusion: Rethinking Polarization</h2>
  <p>The conventional narrative says social media platforms push everyone toward extremes. Our data tells a different story:</p>

  <div class="highlight-box">
    <strong>Reddit isn't becoming a single polarized warzone. It's becoming a collection of isolated villages.</strong>
  </div>

  <p>Event-related communities are getting <em>denser</em> and more self-referential. But the vast majority of Reddit, the gaming, sports, cooking, and hobby communities, is essentially <strong>inoculated</strong> from this noise.</p>

  <blockquote>
    <strong>Final thought:</strong> We may be overestimating polarization by focusing on the loud minority, while ignoring the massive, quiet ecosystem where people just talk about their hobbies.
  </blockquote>
</section>

<hr>

<div class="footer-links">
  <a href="https://github.com/epfl-ada/ada-2025-project-papayarules/blob/main/results.ipynb" class="button">View the Jupyter Notebook</a>
  <a href="https://github.com/epfl-ada/ada-2025-project-papayarules" class="button">GitHub Repository</a>
  <a href="map.html" class="button">Open the Explorer</a>

  <p class="footer-note">
    <em>EPFL Applied Data Analysis - Fall 2025</em>
  </p>
</div>

</div>

<!-- Shared map code click-to-rail driver -->
<script src="assets/js/map_core.js"></script>
<script src="assets/js/map_click_rail.js"></script>

