const API = "/api/v1";

const state = {
  ready: false,
  modelsReady: true,
  activeJob: null,
  view: "photos",
  page: 1,
  perPage: 48,
  total: 0,
  people: [],
  similarTo: null,        // {image_id, filename} when browsing similar photos
  currentPerson: null,    // person shown in the person modal
  selectedFaces: new Set(),
  forgetArmed: false,
  mergeArmedId: null,
};

const RECENT_KEY = "photolib.recentSearches";
const DISMISSED_KEY = "photolib.dismissedMerges";
const RECENT_LIMIT = 8;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const type = response.headers.get("content-type") || "";
  const body = type.includes("json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof body === "object" ? body.detail || JSON.stringify(body) : body;
    throw new Error(message || `Request failed (${response.status})`);
  }
  return body;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setStatus(label, kind = "") {
  $("appStatus").textContent = label;
  $("statusDot").className = `dot ${kind}`;
}

function showError(message) {
  $("errorText").textContent = message;
  $("errorPanel").classList.remove("hidden");
}

function clearError() {
  $("errorPanel").classList.add("hidden");
}

function formatBytes(bytes) {
  if (!bytes) return "0 MB";
  return `${Math.round(bytes / 1024 / 1024)} MB`;
}

function formatDate(value) {
  if (!value) return "Date unknown";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "Date unknown"
    : date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function personLabel(person) {
  return person?.name || `Person ${person?.person_id ?? "?"}`;
}

function personById(personId) {
  return state.people.find((p) => p.person_id === Number(personId)) || null;
}

function faceCropUrl(faceId) {
  return `${API}/faces/${faceId}/crop`;
}

// ---------------------------------------------------------------------------
// Setup, models, jobs
// ---------------------------------------------------------------------------

async function refreshHealth() {
  const health = await request("/health");
  state.ready = Boolean(health.ready);
  $("librarySection").classList.toggle("hidden", !state.ready);
  $("welcomeTitle").innerHTML = state.ready
    ? "Your moments, <span>easy to find.</span>"
    : "Make every photo <span>findable.</span>";
  $("welcomeCopy").textContent = state.ready
    ? "Search in plain language, browse with filters, and put names to the people in your photos. Everything stays on this computer."
    : "Choose the folder where your photos live. Everything is indexed locally and your originals never leave this computer.";
  $("manageTitle").textContent = state.ready ? "Add or rescan photos" : "Choose your photo folder";
  $("indexButton").textContent = state.ready ? "Scan folder" : "Build my library";
  setStatus(state.ready ? "Library ready" : "Setup needed", state.ready ? "ok" : "");
  return health;
}

async function refreshModels() {
  const result = await request("/admin/models");
  state.modelsReady = Boolean(result.ready);
  const missing = (result.models || []).filter((model) => !model.installed);
  const faceModel = missing.find((model) => model.kind === "face");
  const panel = $("modelPanel");
  if (!faceModel) {
    panel.classList.add("hidden");
    $("indexButton").disabled = Boolean(state.activeJob);
    return;
  }

  panel.classList.remove("hidden");
  $("modelTitle").textContent = "Face recognition needs one optional download";
  $("modelCopy").textContent =
    `${faceModel.name} is ${formatBytes(faceModel.approx_bytes)} and is published under: ${faceModel.licence}. ` +
    "It is downloaded only when you approve it. Your photos are never uploaded.";
  $("modelButton").disabled = Boolean(state.activeJob) || Boolean(result.offline);
  $("modelButton").textContent = result.offline ? "Offline mode enabled" : "Download model";
  $("indexButton").disabled = true;
}

async function chooseFolder() {
  clearError();
  $("browseButton").disabled = true;
  $("browseButton").textContent = "Opening…";
  try {
    const result = await request("/admin/select-folder", { method: "POST" });
    if (result.path) {
      $("folderPath").value = result.path;
      localStorage.setItem("photolib.lastFolder", result.path);
    } else if (!result.cancelled && result.detail) {
      showError(result.detail);
    }
  } catch (error) {
    showError(`Could not open the folder picker: ${error.message}. You can type the full path instead.`);
  } finally {
    $("browseButton").disabled = false;
    $("browseButton").textContent = "Browse…";
  }
}

async function startIndex() {
  clearError();
  const folder = $("folderPath").value.trim();
  if (!folder) {
    showError("Choose a folder containing photos first.");
    $("folderPath").focus();
    return;
  }
  localStorage.setItem("photolib.lastFolder", folder);
  try {
    const job = await request("/admin/index", {
      method: "POST",
      body: JSON.stringify({
        folder,
        rebuild: false,
        prune_missing: $("pruneMissing").checked,
      }),
    });
    monitorJob(job);
  } catch (error) {
    showError(`Could not start indexing: ${error.message}`);
  }
}

async function fetchModels() {
  clearError();
  try {
    const job = await request("/admin/models/fetch", { method: "POST" });
    monitorJob(job);
  } catch (error) {
    showError(`Could not download the model: ${error.message}`);
  }
}

function renderJob(job) {
  const panel = $("jobPanel");
  panel.classList.remove("hidden");
  $("jobTitle").textContent = job.kind === "fetch_models" ? "Preparing face recognition" : "Indexing your photos";
  $("jobPhase").textContent = job.phase || job.status;
  $("jobPercent").textContent = job.total ? `${job.percent}%` : "Working…";
  $("progressBar").style.width = job.total ? `${Math.max(1, job.percent)}%` : "12%";
  const counts = job.total ? `${job.current.toLocaleString()} of ${job.total.toLocaleString()}` : "";
  $("jobDetail").textContent = [counts, job.detail?.file || job.detail?.model || ""].filter(Boolean).join(" · ");
  $("cancelJob").classList.toggle("hidden", !["pending", "running"].includes(job.status));
}

async function monitorJob(initial) {
  state.activeJob = initial;
  $("indexButton").disabled = true;
  $("modelButton").disabled = true;
  setStatus("Working", "busy");
  renderJob(initial);

  let job = initial;
  while (["pending", "running"].includes(job.status)) {
    await sleep(650);
    try {
      job = await request(`/admin/jobs/${job.id}`);
      state.activeJob = job;
      renderJob(job);
    } catch (error) {
      showError(`Lost contact with the background job: ${error.message}`);
      break;
    }
  }

  state.activeJob = null;
  if (job.status === "failed") {
    showError(job.error || "The background job failed. See the local photolib log for details.");
  } else if (job.status === "done") {
    $("jobTitle").textContent = job.kind === "fetch_models" ? "Model ready" : "Library updated";
    $("jobPhase").textContent = "done";
    $("jobPercent").textContent = "100%";
    $("progressBar").style.width = "100%";
    $("jobDetail").textContent = summarizeResult(job.result);
    await refreshModels();
    await refreshHealth();
    if (state.ready) {
      await loadLibrary();
    }
  }
  $("indexButton").disabled = !state.modelsReady;
  setStatus(state.ready ? "Library ready" : "Setup needed", state.ready ? "ok" : "");
}

function summarizeResult(result) {
  if (!result) return "Finished successfully.";
  const parts = [];
  for (const key of ["added", "updated", "skipped", "removed"]) {
    if (Number.isFinite(result[key])) parts.push(`${result[key]} ${key}`);
  }
  return parts.length ? parts.join(" · ") : "Finished successfully.";
}

async function findActiveJob() {
  const jobs = await request("/admin/jobs");
  const active = jobs.find((job) => ["pending", "running"].includes(job.status));
  if (active) monitorJob(active);
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

function setView(view) {
  state.view = view;
  $("photosView").classList.toggle("hidden", view !== "photos");
  $("peopleView").classList.toggle("hidden", view !== "people");
  $("tabPhotos").classList.toggle("active", view === "photos");
  $("tabPeople").classList.toggle("active", view === "people");
  $("tabPhotos").setAttribute("aria-selected", String(view === "photos"));
  $("tabPeople").setAttribute("aria-selected", String(view === "people"));
  if (view === "people") loadPeopleView();
}

// ---------------------------------------------------------------------------
// Search, filters, history
// ---------------------------------------------------------------------------

function recentSearches() {
  try {
    const parsed = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function rememberSearch(query) {
  if (!query) return;
  const next = [query, ...recentSearches().filter((q) => q !== query)].slice(0, RECENT_LIMIT);
  localStorage.setItem(RECENT_KEY, JSON.stringify(next));
  renderRecentSearches();
}

function renderRecentSearches() {
  const box = $("recentSearches");
  const recents = recentSearches();
  if (!recents.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  box.innerHTML = recents.map((q) =>
    `<button class="chip" type="button" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`
  ).join("") +
    '<button class="chip ghost" type="button" data-clear="1">Clear history</button>';
  box.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (chip.dataset.clear) {
        localStorage.removeItem(RECENT_KEY);
        renderRecentSearches();
        return;
      }
      $("searchInput").value = chip.dataset.query;
      search(1);
    });
  });
}

function currentFilters() {
  const from = $("dateFrom").value;
  const to = $("dateTo").value;
  const person = $("personFilter").value;
  return {
    start_date: from ? `${from}T00:00:00` : null,
    end_date: to ? `${to}T23:59:59` : null,
    people_ids: person ? [Number(person)] : [],
  };
}

function clearFilters() {
  $("searchInput").value = "";
  $("dateFrom").value = "";
  $("dateTo").value = "";
  $("personFilter").value = "";
  $("sortSelect").value = "relevance";
  state.similarTo = null;
  $("similarBanner").classList.add("hidden");
  search(1);
}

async function search(page = 1) {
  clearError();
  state.page = page;
  state.similarTo = null;
  $("similarBanner").classList.add("hidden");
  $("photoGrid").innerHTML = '<div class="empty">Loading your library…</div>';
  const query = $("searchInput").value.trim();
  const filters = currentFilters();
  try {
    const result = await request("/search", {
      method: "POST",
      body: JSON.stringify({
        query: query || null,
        ...filters,
        sort: $("sortSelect").value,
        page,
        per_page: state.perPage,
      }),
    });
    state.total = result.total;
    rememberSearch(query);
    renderPhotos(result);
    const took = result.took_ms ? ` · ${Math.round(result.took_ms)} ms` : "";
    $("resultCount").textContent =
      `${result.total.toLocaleString()} ${result.total === 1 ? "photo" : "photos"}${took}`;
  } catch (error) {
    $("photoGrid").innerHTML = "";
    showError(`Search failed: ${error.message}`);
  }
}

async function showSimilar(imageId, filename) {
  clearError();
  setView("photos");
  $("photoGrid").innerHTML = '<div class="empty">Finding similar photos…</div>';
  try {
    const body = await request(`/images/${imageId}/similar?limit=48`);
    state.similarTo = { image_id: imageId, filename };
    $("similarText").textContent = `Photos that look like ${filename || "the selected photo"}.`;
    $("similarBanner").classList.remove("hidden");
    renderPhotos({ results: body.results, total: body.results.length, page: 1, per_page: state.perPage });
    $("resultCount").textContent =
      `${body.results.length.toLocaleString()} similar ${body.results.length === 1 ? "photo" : "photos"}`;
    $("pager").classList.add("hidden");
  } catch (error) {
    $("photoGrid").innerHTML = "";
    showError(`Similar search failed: ${error.message}`);
  }
}

function browsePerson(personId) {
  setView("photos");
  closePerson();
  closePhoto();
  $("searchInput").value = "";
  $("personFilter").value = String(personId);
  $("sortSelect").value = "date_desc";
  search(1);
  $("photosView").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderPhotos(result) {
  const grid = $("photoGrid");
  if (!result.results.length) {
    grid.innerHTML = '<div class="empty">No photos matched. Try a broader description, widen the dates, or rescan your folder.</div>';
  } else {
    grid.innerHTML = result.results.map((photo) => `
      <button class="photo" data-image-id="${photo.image_id}" data-filename="${escapeHtml(photo.filename || "")}" aria-label="Open ${escapeHtml(photo.filename || "photo")}">
        <img loading="lazy" src="${API}/images/${photo.image_id}/thumbnail?size=grid&format=webp" alt="${escapeHtml(photo.filename || "Photo")}">
        <span class="photo-meta">
          <span>${escapeHtml(formatDate(photo.taken_at))}</span>
          <span>${photo.face_count ? `${photo.face_count} ${photo.face_count === 1 ? "face" : "faces"}` : ""}</span>
        </span>
        ${typeof photo.score === "number" ? `<span class="score" title="Match strength">${Math.round(photo.score * 100)}%</span>` : ""}
      </button>
    `).join("");
    grid.querySelectorAll(".photo").forEach((button) => {
      button.addEventListener("click", () =>
        openPhoto(Number(button.dataset.imageId), button.dataset.filename));
    });
  }
  const pages = Math.max(1, Math.ceil(result.total / result.per_page));
  $("pageLabel").textContent = `Page ${result.page} of ${pages}`;
  $("prevPage").disabled = result.page <= 1;
  $("nextPage").disabled = result.page >= pages;
  $("pager").classList.toggle("hidden", result.total <= result.per_page);
}

// ---------------------------------------------------------------------------
// Photo modal
// ---------------------------------------------------------------------------

async function openPhoto(imageId, filename = "") {
  $("photoModal").classList.remove("hidden");
  $("modalImage").src = `${API}/images/${imageId}`;
  $("modalName").textContent = filename || "Loading…";
  $("modalMeta").textContent = "";
  $("modalPeople").innerHTML = "";
  $("modalSimilar").onclick = () => {
    closePhoto();
    showSimilar(imageId, filename);
  };
  try {
    const details = await request(`/images/${imageId}/details`);
    $("modalName").textContent = details.filename || "Photo";
    $("modalMeta").textContent = [
      formatDate(details.taken_at),
      details.camera,
      details.place,
      details.width && details.height ? `${details.width}×${details.height}` : "",
    ].filter(Boolean).join(" · ");
    const people = details.people || [];
    $("modalPeople").innerHTML = people.map((p) => {
      const known = personById(p.person_id);
      const label = p.name || known?.name || `Person ${p.person_id}`;
      return `<button class="chip" type="button" data-person-id="${p.person_id}">${escapeHtml(label)}</button>`;
    }).join("");
    $("modalPeople").querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => browsePerson(Number(chip.dataset.personId)));
    });
  } catch {
    $("modalName").textContent = filename || "Photo";
  }
}

function closePhoto() {
  $("photoModal").classList.add("hidden");
  $("modalImage").removeAttribute("src");
}

// ---------------------------------------------------------------------------
// People view
// ---------------------------------------------------------------------------

async function loadPeople() {
  try {
    state.people = await request("/people?min_photos=1");
    const select = $("personFilter");
    const current = select.value;
    select.innerHTML = '<option value="">All people</option>' +
      state.people.map((person) =>
        `<option value="${person.person_id}">${escapeHtml(personLabel(person))} (${person.photo_count})</option>`
      ).join("");
    if ([...select.options].some((o) => o.value === current)) select.value = current;
  } catch {
    state.people = [];
  }
}

function dismissedMerges() {
  try {
    const parsed = JSON.parse(localStorage.getItem(DISMISSED_KEY) || "[]");
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function mergeKey(a, b) {
  return `${Math.min(a, b)}:${Math.max(a, b)}`;
}

function dismissMerge(a, b) {
  const set = dismissedMerges();
  set.add(mergeKey(a, b));
  localStorage.setItem(DISMISSED_KEY, JSON.stringify([...set]));
}

function coverHtml(person, size = "") {
  if (person && person.cover_face_id >= 0) {
    return `<img class="face-img ${size}" loading="lazy" src="${faceCropUrl(person.cover_face_id)}" alt="">`;
  }
  const initial = (person?.name || "?").trim().charAt(0).toUpperCase() || "?";
  return `<span class="face-img placeholder ${size}">${escapeHtml(initial)}</span>`;
}

async function loadPeopleView() {
  await loadPeople();
  renderPeopleGrid();
  loadMergeSuggestions();
}

function renderPeopleGrid() {
  const grid = $("peopleGrid");
  if (!state.people.length) {
    grid.innerHTML = '<div class="empty">No people found yet. Index a folder with photos of people, then check back here.</div>';
    return;
  }
  grid.innerHTML = state.people.map((person) => `
    <button class="person-tile" type="button" data-person-id="${person.person_id}">
      ${coverHtml(person, "lg")}
      <span class="person-tile-name">${escapeHtml(personLabel(person))}</span>
      <span class="person-tile-count">${person.photo_count} ${person.photo_count === 1 ? "photo" : "photos"}</span>
    </button>
  `).join("");
  grid.querySelectorAll(".person-tile").forEach((tile) => {
    tile.addEventListener("click", () => openPerson(Number(tile.dataset.personId)));
  });
}

async function loadMergeSuggestions() {
  const panel = $("mergeSuggestPanel");
  const list = $("mergeSuggestList");
  try {
    const body = await request("/people/merge-suggestions?limit=30");
    const dismissed = dismissedMerges();
    const pairs = (body.suggestions || []).filter(
      (s) => !dismissed.has(mergeKey(s.source.person_id, s.target.person_id)));
    if (!pairs.length) {
      panel.classList.add("hidden");
      list.innerHTML = "";
      return;
    }
    panel.classList.remove("hidden");
    list.innerHTML = pairs.map((s, i) => `
      <div class="merge-card" data-index="${i}">
        <div class="merge-faces">
          ${coverHtml(s.source)}
          <span class="merge-arrow" aria-hidden="true">→</span>
          ${coverHtml(s.target)}
        </div>
        <div class="merge-copy">
          <strong>${escapeHtml(personLabel(s.source))}</strong> looks like
          <strong>${escapeHtml(personLabel(s.target))}</strong>
          <span class="merge-sim">${Math.round(s.similarity * 100)}% similar</span>
        </div>
        <div class="merge-actions">
          <button class="btn primary slim-btn" type="button" data-merge="${i}">Merge</button>
          <button class="btn ghost slim-btn" type="button" data-dismiss="${i}">Not the same</button>
        </div>
      </div>
    `).join("");
    list.querySelectorAll("[data-merge]").forEach((button) => {
      button.addEventListener("click", async () => {
        const s = pairs[Number(button.dataset.merge)];
        button.disabled = true;
        button.textContent = "Merging…";
        try {
          await mergePeople(s.source.person_id, s.target.person_id);
        } catch (error) {
          showError(`Merge failed: ${error.message}`);
          button.disabled = false;
          button.textContent = "Merge";
        }
      });
    });
    list.querySelectorAll("[data-dismiss]").forEach((button) => {
      button.addEventListener("click", () => {
        const s = pairs[Number(button.dataset.dismiss)];
        dismissMerge(s.source.person_id, s.target.person_id);
        loadMergeSuggestions();
      });
    });
  } catch {
    panel.classList.add("hidden");
  }
}

async function mergePeople(sourceId, targetId) {
  await request("/people/merge", {
    method: "POST",
    body: JSON.stringify({ source_id: sourceId, target_id: targetId }),
  });
  await Promise.all([loadStats(), loadPeople()]);
  renderPeopleGrid();
  loadMergeSuggestions();
}

// ---------------------------------------------------------------------------
// Person modal
// ---------------------------------------------------------------------------

async function openPerson(personId) {
  const person = personById(personId);
  if (!person) return;
  state.currentPerson = person;
  state.selectedFaces = new Set();
  state.forgetArmed = false;
  state.mergeArmedId = null;

  $("personModal").classList.remove("hidden");
  $("personName").value = person.name || "";
  $("personCover").innerHTML = coverHtml(person, "xl");
  $("personMeta").textContent =
    `${person.photo_count} ${person.photo_count === 1 ? "photo" : "photos"} · ${person.face_count} ${person.face_count === 1 ? "face" : "faces"}`;
  $("personHide").textContent = person.hidden ? "Unhide" : "Hide";
  $("personForget").textContent = "Forget person";
  $("mergePicker").classList.add("hidden");
  $("mergeFilter").value = "";
  $("detachSelected").disabled = true;
  $("personFaces").innerHTML = '<div class="empty slim-empty">Loading faces…</div>';
  $("personSuggestSection").classList.add("hidden");

  loadPersonFaces(personId);
  loadPersonSuggestions(personId);
}

function closePerson() {
  $("personModal").classList.add("hidden");
  state.currentPerson = null;
}

async function loadPersonFaces(personId) {
  try {
    const body = await request(`/people/${personId}/faces?limit=200`);
    const faces = body.faces || [];
    $("personFacesTitle").textContent = `Faces (${faces.length})`;
    if (!faces.length) {
      $("personFaces").innerHTML = '<div class="empty slim-empty">No faces recorded.</div>';
      return;
    }
    $("personFaces").innerHTML = faces.map((f) => `
      <button class="face-tile" type="button" data-face-id="${f.face_id}" title="Quality ${Math.round((f.quality || 0) * 100)}%${f.confirmed ? " · confirmed by you" : ""}">
        <img class="face-img" loading="lazy" src="${faceCropUrl(f.face_id)}" alt="">
        ${f.confirmed ? '<span class="face-badge" aria-label="Confirmed">✓</span>' : ""}
      </button>
    `).join("");
    $("personFaces").querySelectorAll(".face-tile").forEach((tile) => {
      tile.addEventListener("click", () => {
        const id = Number(tile.dataset.faceId);
        if (state.selectedFaces.has(id)) {
          state.selectedFaces.delete(id);
          tile.classList.remove("selected");
        } else {
          state.selectedFaces.add(id);
          tile.classList.add("selected");
        }
        $("detachSelected").disabled = !state.selectedFaces.size;
        $("detachSelected").textContent = state.selectedFaces.size
          ? `Not this person (${state.selectedFaces.size})`
          : "Not this person";
      });
    });
  } catch (error) {
    $("personFaces").innerHTML = `<div class="empty slim-empty">Could not load faces: ${escapeHtml(error.message)}</div>`;
  }
}

async function loadPersonSuggestions(personId) {
  try {
    const body = await request(`/people/${personId}/suggestions?limit=30`);
    const suggestions = body.suggestions || [];
    if (!suggestions.length) return;
    $("personSuggestSection").classList.remove("hidden");
    $("personSuggestions").innerHTML = suggestions.map((s) => `
      <div class="face-tile suggest" data-face-id="${s.face_id}" title="${Math.round(s.similarity * 100)}% similar">
        <img class="face-img" loading="lazy" src="${faceCropUrl(s.face_id)}" alt="">
        <span class="suggest-actions">
          <button class="mini-btn yes" type="button" data-accept="${s.face_id}" aria-label="Yes, this is them">✓</button>
          <button class="mini-btn no" type="button" data-reject="${s.face_id}" aria-label="No, not them">✗</button>
        </span>
      </div>
    `).join("");
    const refresh = async () => {
      await Promise.all([loadStats(), loadPeople()]);
      const person = personById(personId);
      if (person && state.currentPerson?.person_id === personId) {
        state.currentPerson = person;
        $("personMeta").textContent =
          `${person.photo_count} ${person.photo_count === 1 ? "photo" : "photos"} · ${person.face_count} ${person.face_count === 1 ? "face" : "faces"}`;
      }
      loadPersonFaces(personId);
      loadPersonSuggestions(personId);
    };
    $("personSuggestions").querySelectorAll("[data-accept]").forEach((button) => {
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await request("/faces/assign", {
            method: "POST",
            body: JSON.stringify({ face_ids: [Number(button.dataset.accept)], person_id: personId }),
          });
          await refresh();
        } catch (error) {
          showError(`Could not add the face: ${error.message}`);
        }
      });
    });
    $("personSuggestions").querySelectorAll("[data-reject]").forEach((button) => {
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await request("/faces/detach", {
            method: "POST",
            body: JSON.stringify({ face_ids: [Number(button.dataset.reject)] }),
          });
          button.closest(".face-tile")?.remove();
        } catch (error) {
          showError(`Could not dismiss the face: ${error.message}`);
        }
      });
    });
  } catch {
    // Suggestions are best-effort decoration; the panel simply stays hidden.
  }
}

async function savePersonName() {
  const person = state.currentPerson;
  if (!person) return;
  const name = $("personName").value.trim();
  if (!name) return;
  try {
    await request(`/people/${person.person_id}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    });
    await loadPeople();
    renderPeopleGrid();
    $("personSave").textContent = "Saved ✓";
    setTimeout(() => { $("personSave").textContent = "Save"; }, 1400);
  } catch (error) {
    showError(`Rename failed: ${error.message}`);
  }
}

function toggleMergePicker() {
  const person = state.currentPerson;
  if (!person) return;
  const picker = $("mergePicker");
  const show = picker.classList.contains("hidden");
  picker.classList.toggle("hidden", !show);
  if (show) {
    renderMergeList("");
    $("mergeFilter").focus();
  }
}

function renderMergeList(filterText) {
  const person = state.currentPerson;
  if (!person) return;
  const needle = filterText.trim().toLowerCase();
  const candidates = state.people.filter((p) =>
    p.person_id !== person.person_id &&
    (!needle || personLabel(p).toLowerCase().includes(needle)));
  $("mergeList").innerHTML = candidates.length
    ? candidates.map((p) => `
        <button class="merge-item ${state.mergeArmedId === p.person_id ? "armed" : ""}" type="button" data-person-id="${p.person_id}">
          ${coverHtml(p)}
          <span class="merge-item-name">${escapeHtml(personLabel(p))}</span>
          <span class="merge-item-count">${p.photo_count} photos</span>
          <span class="merge-item-cta">${state.mergeArmedId === p.person_id ? "Click again to merge" : ""}</span>
        </button>
      `).join("")
    : '<div class="empty slim-empty">No other people match.</div>';
  $("mergeList").querySelectorAll(".merge-item").forEach((item) => {
    item.addEventListener("click", async () => {
      const targetId = Number(item.dataset.personId);
      if (state.mergeArmedId !== targetId) {
        state.mergeArmedId = targetId;
        renderMergeList($("mergeFilter").value);
        return;
      }
      try {
        await mergePeople(person.person_id, targetId);
        closePerson();
        const target = personById(targetId);
        if (target) openPerson(targetId);
      } catch (error) {
        showError(`Merge failed: ${error.message}`);
      }
    });
  });
}

async function toggleHidden() {
  const person = state.currentPerson;
  if (!person) return;
  try {
    await request(`/people/${person.person_id}/hidden`, {
      method: "POST",
      body: JSON.stringify({ hidden: !person.hidden }),
    });
    await loadPeople();
    renderPeopleGrid();
    closePerson();
  } catch (error) {
    showError(`Could not update the person: ${error.message}`);
  }
}

async function forgetPerson() {
  const person = state.currentPerson;
  if (!person) return;
  if (!state.forgetArmed) {
    state.forgetArmed = true;
    $("personForget").textContent = "Really forget? Faces go back to review";
    return;
  }
  try {
    await request(`/people/${person.person_id}`, { method: "DELETE" });
    await Promise.all([loadStats(), loadPeople()]);
    renderPeopleGrid();
    loadMergeSuggestions();
    closePerson();
  } catch (error) {
    showError(`Could not forget the person: ${error.message}`);
  }
}

async function detachSelectedFaces() {
  const person = state.currentPerson;
  if (!person || !state.selectedFaces.size) return;
  const ids = [...state.selectedFaces];
  $("detachSelected").disabled = true;
  try {
    await request("/faces/detach", {
      method: "POST",
      body: JSON.stringify({ face_ids: ids }),
    });
    state.selectedFaces = new Set();
    $("detachSelected").textContent = "Not this person";
    await Promise.all([loadStats(), loadPeople()]);
    renderPeopleGrid();
    loadPersonFaces(person.person_id);
  } catch (error) {
    showError(`Could not detach the faces: ${error.message}`);
    $("detachSelected").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Stats and startup
// ---------------------------------------------------------------------------

async function loadStats() {
  const stats = await request("/stats");
  $("photoCount").textContent = (stats.total_images || 0).toLocaleString();
  $("peopleCount").textContent = (stats.total_people || 0).toLocaleString();
  $("faceCount").textContent = (stats.total_faces || 0).toLocaleString();
}

async function loadLibrary() {
  await Promise.all([loadPeople(), loadStats()]);
  renderRecentSearches();
  await search(1);
  if (state.view === "people") loadPeopleView();
}

async function cancelJob() {
  if (!state.activeJob) return;
  try {
    await request(`/admin/jobs/${state.activeJob.id}`, { method: "DELETE" });
    $("cancelJob").disabled = true;
    $("cancelJob").textContent = "Cancelling…";
  } catch (error) {
    showError(`Could not cancel the job: ${error.message}`);
  }
}

async function init() {
  $("folderPath").value = localStorage.getItem("photolib.lastFolder") || "";
  $("browseButton").addEventListener("click", chooseFolder);
  $("indexButton").addEventListener("click", startIndex);
  $("modelButton").addEventListener("click", fetchModels);
  $("cancelJob").addEventListener("click", cancelJob);
  $("dismissError").addEventListener("click", clearError);

  $("tabPhotos").addEventListener("click", () => setView("photos"));
  $("tabPeople").addEventListener("click", () => setView("people"));

  $("searchForm").addEventListener("submit", (event) => {
    event.preventDefault();
    search(1);
  });
  $("sortSelect").addEventListener("change", () => search(1));
  $("personFilter").addEventListener("change", () => search(1));
  $("dateFrom").addEventListener("change", () => search(1));
  $("dateTo").addEventListener("change", () => search(1));
  $("clearFilters").addEventListener("click", clearFilters);
  $("similarClear").addEventListener("click", () => {
    state.similarTo = null;
    $("similarBanner").classList.add("hidden");
    search(1);
  });
  $("prevPage").addEventListener("click", () => search(state.page - 1));
  $("nextPage").addEventListener("click", () => search(state.page + 1));

  $("modalClose").addEventListener("click", closePhoto);
  $("photoModal").addEventListener("click", (event) => {
    if (event.target === $("photoModal")) closePhoto();
  });

  $("personClose").addEventListener("click", closePerson);
  $("personModal").addEventListener("click", (event) => {
    if (event.target === $("personModal")) closePerson();
  });
  $("personSave").addEventListener("click", savePersonName);
  $("personName").addEventListener("keydown", (event) => {
    if (event.key === "Enter") savePersonName();
  });
  $("personPhotos").addEventListener("click", () => {
    if (state.currentPerson) browsePerson(state.currentPerson.person_id);
  });
  $("personMergeBtn").addEventListener("click", toggleMergePicker);
  $("mergeFilter").addEventListener("input", () => {
    state.mergeArmedId = null;
    renderMergeList($("mergeFilter").value);
  });
  $("personHide").addEventListener("click", toggleHidden);
  $("personForget").addEventListener("click", forgetPerson);
  $("detachSelected").addEventListener("click", detachSelectedFaces);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closePhoto();
      closePerson();
    }
  });

  try {
    await refreshHealth();
    await refreshModels();
    await findActiveJob();
    if (state.ready) await loadLibrary();
  } catch (error) {
    setStatus("Needs attention");
    showError(`photolib could not finish starting: ${error.message}`);
  }
}

document.addEventListener("DOMContentLoaded", init);
