/* ===================================================================
   NeuroSearch — frontend logic
   Talks to the FastAPI backend, renders results, and draws the
   keyword-vs-semantic score breakdown for every result.
   =================================================================== */

const API = "";  // same origin — FastAPI serves this page

// ------------------------- ELEMENTS -------------------------

const els = {
  query:        document.getElementById("query"),
  searchBtn:    document.getElementById("searchBtn"),
  results:      document.getElementById("results"),
  status:       document.getElementById("status"),
  alpha:        document.getElementById("alpha"),
  sliderBlock:  document.getElementById("sliderBlock"),
  keywordPct:   document.getElementById("keywordPct"),
  semanticPct:  document.getElementById("semanticPct"),
  modeButtons:  document.querySelectorAll(".mode-btn"),
};

// ------------------------- STATE -------------------------

const state = {
  mode: "hybrid",
  alpha: 0.5,
  lastQuery: "",
  indexReady: false,
};

// ------------------------- HELPERS -------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/** Highlight query words inside the snippet so matches are visible. */
function highlight(text, query) {
  const safe = escapeHtml(text);
  const words = query
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 2)
    .map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));

  if (words.length === 0) return safe;
  return safe.replace(new RegExp(`\\b(${words.join("|")})`, "gi"), "<mark>$1</mark>");
}

function setStatus(html) {
  els.status.innerHTML = html;
}

function showMessage(heading, body) {
  els.results.innerHTML = `
    <div class="message">
      <h2>${heading}</h2>
      <p>${body}</p>
    </div>`;
}

function showSkeletons(count = 4) {
  els.results.innerHTML = Array.from(
    { length: count },
    () => `<div class="skeleton"></div>`
  ).join("");
}

// ------------------------- RENDERING -------------------------

function renderResults(data) {
  if (data.results.length === 0) {
    showMessage(
      "No matches",
      `Nothing in the index matches &ldquo;${escapeHtml(data.query)}&rdquo;. Try broader wording, or switch to Semantic mode to search by meaning.`
    );
    return;
  }

  els.results.innerHTML = data.results
    .map((r) => {
      // Split the bar proportionally to how much each signal contributed
      const total = r.keyword_score + r.semantic_score;
      const keywordWidth = total > 0 ? (r.keyword_score / total) * 100 : 50;
      const semanticWidth = 100 - keywordWidth;

      return `
      <article class="result">
        <div class="result-head">
          <span class="result-rank">${r.rank}</span>
          <a class="result-title" href="${escapeHtml(r.url)}" target="_blank" rel="noopener">
            ${escapeHtml(r.title)}
          </a>
        </div>
        <span class="result-url">${escapeHtml(r.url)}</span>
        <p class="result-snippet">${highlight(r.snippet, data.query)}</p>

        <div class="score-bar" title="How this result's score was made up">
          <div class="score-seg score-seg-keyword" style="width:${keywordWidth}%"></div>
          <div class="score-seg score-seg-semantic" style="width:${semanticWidth}%"></div>
        </div>
        <div class="score-legend">
          <span><i class="dot dot-keyword"></i>keyword <b>${r.keyword_score.toFixed(3)}</b></span>
          <span><i class="dot dot-semantic"></i>semantic <b>${r.semantic_score.toFixed(3)}</b></span>
          <span>final <b>${r.score.toFixed(3)}</b></span>
        </div>
      </article>`;
    })
    .join("");
}

// ------------------------- SEARCH -------------------------

async function runSearch() {
  const query = els.query.value.trim();

  if (!query) {
    els.query.focus();
    return;
  }

  if (!state.indexReady) {
    showMessage(
      "Index not built yet",
      `Run <code>python crawler/crawler.py</code> then <code>python indexer/indexer.py</code>, and restart the server.`
    );
    return;
  }

  state.lastQuery = query;
  els.searchBtn.disabled = true;
  setStatus("Searching…");
  showSkeletons();

  const params = new URLSearchParams({
    q: query,
    mode: state.mode,
    alpha: state.alpha,
    limit: 10,
  });

  try {
    const response = await fetch(`${API}/api/search?${params}`);
    if (!response.ok) throw new Error(`Server returned ${response.status}`);

    const data = await response.json();
    renderResults(data);
    setStatus(
      `<b>${data.count}</b> results · <b>${data.time_ms}ms</b> · ${data.mode} ranking`
    );
  } catch (err) {
    showMessage(
      "Search failed",
      `Couldn't reach the API. Make sure the server is running: <code>python app/main.py</code>`
    );
    setStatus("Connection error");
    console.error(err);
  } finally {
    els.searchBtn.disabled = false;
  }
}

/** Re-run the last search — used when the mix changes. */
function rerunIfSearched() {
  if (state.lastQuery) runSearch();
}

// ------------------------- CONTROLS -------------------------

function setMode(mode) {
  state.mode = mode;
  els.modeButtons.forEach((btn) =>
    btn.classList.toggle("is-active", btn.dataset.mode === mode)
  );
  // The slider only means something in hybrid mode
  els.sliderBlock.classList.toggle("is-off", mode !== "hybrid");
  rerunIfSearched();
}

function setAlpha(percent) {
  state.alpha = percent / 100;
  els.semanticPct.textContent = `${percent}%`;
  els.keywordPct.textContent = `${100 - percent}%`;
}

// ------------------------- STARTUP -------------------------

async function checkIndex() {
  try {
    const response = await fetch(`${API}/api/stats`);
    const stats = await response.json();

    if (stats.ready) {
      state.indexReady = true;
      setStatus(
        `<b>${stats.documents}</b> documents indexed · <b>${stats.embedding_dimensions}</b>-dim embeddings · ${stats.model}`
      );
      showMessage(
        "Ready to search",
        "Try a question in plain English. Slide the mix towards neural semantic to see meaning-based matching take over from exact keywords."
      );
    } else {
      setStatus("Index not built");
      showMessage(
        "Index not built yet",
        `Run <code>python crawler/crawler.py</code> then <code>python indexer/indexer.py</code>, and restart the server.`
      );
    }
  } catch (err) {
    setStatus("Backend offline");
    showMessage(
      "Backend not running",
      `Start the server with <code>python app/main.py</code> and reload this page.`
    );
  }
}

// ------------------------- EVENTS -------------------------

els.searchBtn.addEventListener("click", runSearch);

els.query.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});

els.modeButtons.forEach((btn) =>
  btn.addEventListener("click", () => setMode(btn.dataset.mode))
);

// Update the labels live while dragging, but only re-search on release
els.alpha.addEventListener("input", (e) => setAlpha(Number(e.target.value)));
els.alpha.addEventListener("change", rerunIfSearched);

// Go
setAlpha(Number(els.alpha.value));
checkIndex();
els.query.focus();
