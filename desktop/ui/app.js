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
  roots: [],
  months: [],
  selectedPeople: [],     // person_ids in the current search
  peopleMode: "all",      // "all" = every selected person must be in the photo
  similarTo: null,
  currentPerson: null,
  selectedFaces: new Set(),
  selectedModalFace: null,
  forgetArmed: false,
  mergeArmedId: null,
  loadCount: 0,
};

const RECENT_KEY = "photolib.recentSearches";
const DISMISSED_KEY = "photolib.dismissedMerges";
const RECENT_LIMIT = 8;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------------------------------------------------------------------------
// Fetch + loading strip
// ---------------------------------------------------------------------------

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

function startLoad() {
  state.loadCount += 1;
  $("loadStrip").classList.add("on");
}

function endLoad() {
  state.loadCount = Math.max(0, state.loadCount - 1);
  if (!state.loadCount) $("loadStrip").classList.remove("on");
}

function skeletonGrid(count = 12) {
  let tiles = "";
  for (let i = 0; i < count; i++) {
    tiles += `<div class="photo skeleton"><span class="frame-no mono">${String(i + 1).padStart(3, "0")}</span></div>`;
  }
  return tiles;
}

function skeletonPeople(count = 10) {
  let tiles = "";
  for (let i = 0; i < count; i++) {
    tiles += '<div class="person-tile skeleton-tile"><span class="face-img lg shimmer"></span><span class="skeleton-line"></span></div>';
  }
  return tiles;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

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
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(bytes > 100 * 1024 * 1024 ? 0 : 1)} MB`;
}

function formatDate(value) {
  if (!value) return "no date";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "no date"
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
  $("setupHero").classList.toggle("hidden", state.ready);
  $("managePanel").classList.toggle("hidden", state.ready);
  setStatus(state.ready ? "Library ready" : "Setup needed", state.ready ? "ok" : "");
  return health;
}

async function refreshModels() {
  const result = await request("/admin/models");
  state.modelsReady = Boolean(result.ready);
  renderModelStatus(result);
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

function renderModelStatus(result) {
  const lines = (result.models || []).map((m) =>
    `<div class="model-line"><span>${escapeHtml(m.name)}</span>` +
    `<span>${m.installed ? "installed" : "not installed"}` +
    `${m.approx_bytes ? ` · ${formatBytes(m.approx_bytes)}` : ""}</span></div>`);
  $("modelStatusList").innerHTML = lines.join("") ||
    '<div class="model-line"><span>No models configured</span></div>';
  $("modelStatusCopy").textContent = result.offline
    ? "Offline mode is on — nothing can be downloaded."
    : "Everything runs on this machine. Nothing is uploaded, ever.";
}

async function chooseFolder(intoField = true) {
  clearError();
  try {
    const result = await request("/admin/select-folder", { method: "POST" });
    if (result.path) {
      if (intoField) {
        $("folderPath").value = result.path;
        localStorage.setItem("photolib.lastFolder", result.path);
      }
      return result.path;
    }
    if (!result.cancelled && result.detail) showError(result.detail);
  } catch (error) {
    showError(`Could not open the folder picker: ${error.message}.`);
  }
  return null;
}

async function startIndex(folder, { monitor = true } = {}) {
  clearError();
  if (!folder) return null;
  try {
    const job = await request("/admin/index", {
      method: "POST",
      body: JSON.stringify({
        folder,
        rebuild: false,
        prune_missing: $("pruneMissing")?.checked || false,
      }),
    });
    if (monitor) return monitorJob(job);
    return job;
  } catch (error) {
    showError(`Could not start indexing: ${error.message}`);
    return null;
  }
}

async function startIndexFromSetup() {
  const folder = $("folderPath").value.trim();
  if (!folder) {
    showError("Choose a folder containing photos first.");
    $("folderPath").focus();
    return;
  }
  localStorage.setItem("photolib.lastFolder", folder);
  await startIndex(folder);
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
    if (state.ready) await loadLibrary();
  }
  $("indexButton").disabled = !state.modelsReady;
  setStatus(state.ready ? "Library ready" : "Setup needed", state.ready ? "ok" : "");
  return job;
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
  for (const [id, name] of [["photosView", "photos"], ["peopleView", "people"], ["manageView", "manage"]]) {
    $(id).classList.toggle("hidden", view !== name);
  }
  for (const [id, name] of [["tabPhotos", "photos"], ["tabPeople", "people"], ["tabManage", "manage"]]) {
    $(id).classList.toggle("active", view === name);
    $(id).setAttribute("aria-selected", String(view === name));
  }
  if (view === "people") loadPeopleView();
  if (view === "manage") loadManageView();
}

// ---------------------------------------------------------------------------
// Search history
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
  box.innerHTML = '<span class="chips-label mono">RECENT</span>' + recents.map((q) =>
    `<button class="chip" type="button" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`
  ).join("") +
    '<button class="chip ghost" type="button" data-clear="1">clear</button>';
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

// ---------------------------------------------------------------------------
// People picker (multi-person search)
// ---------------------------------------------------------------------------

function togglePeoplePanel(show) {
  const panel = $("peoplePanel");
  const willShow = show ?? panel.classList.contains("hidden");
  panel.classList.toggle("hidden", !willShow);
  $("peopleButton").setAttribute("aria-expanded", String(willShow));
  if (willShow) {
    $("peopleSearch").value = "";
    renderPeopleOptions("");
    $("peopleSearch").focus();
  }
}

function renderPeopleOptions(filterText) {
  const needle = filterText.trim().toLowerCase();
  const options = state.people.filter((p) =>
    !needle || personLabel(p).toLowerCase().includes(needle));
  $("peopleOptions").innerHTML = options.length
    ? options.map((p) => {
        const selected = state.selectedPeople.includes(p.person_id);
        return `<button class="picker-row ${selected ? "selected" : ""}" type="button" data-person-id="${p.person_id}">
          ${coverHtml(p)}
          <span class="picker-name">${escapeHtml(personLabel(p))}</span>
          <span class="picker-count mono">${p.photo_count}</span>
          <span class="picker-check">${selected ? "✓" : ""}</span>
        </button>`;
      }).join("")
    : '<div class="empty slim-empty">No people match.</div>';
  $("peopleOptions").querySelectorAll(".picker-row").forEach((row) => {
    row.addEventListener("click", () => {
      const id = Number(row.dataset.personId);
      const at = state.selectedPeople.indexOf(id);
      if (at >= 0) state.selectedPeople.splice(at, 1);
      else state.selectedPeople.push(id);
      renderPeopleOptions($("peopleSearch").value);
      renderSelectedPeople();
      search(1, { keepPanel: true });
    });
  });
}

function renderSelectedPeople() {
  const box = $("selectedPeople");
  const count = state.selectedPeople.length;
  $("peopleButton").textContent = count ? `People (${count})` : "People";
  $("peopleButton").classList.toggle("active-filter", count > 0);
  if (!count) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  const modeNote = count > 1
    ? `<span class="chips-label mono">${state.peopleMode === "all" ? "ALL OF" : "ANY OF"}</span>` : "";
  box.innerHTML = modeNote + state.selectedPeople.map((id) => {
    const p = personById(id);
    return `<button class="chip person-chip" type="button" data-person-id="${id}" title="Remove from search">
      ${escapeHtml(p ? personLabel(p) : `Person ${id}`)} ×</button>`;
  }).join("");
  box.querySelectorAll(".person-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      state.selectedPeople = state.selectedPeople.filter(
        (id) => id !== Number(chip.dataset.personId));
      renderSelectedPeople();
      search(1);
    });
  });
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

function currentFilters() {
  const from = $("dateFrom").value;
  const to = $("dateTo").value;
  return {
    start_date: from ? `${from}T00:00:00` : null,
    end_date: to ? `${to}T23:59:59` : null,
    people_ids: state.selectedPeople,
    people_mode: state.selectedPeople.length > 1 ? state.peopleMode : "any",
    camera: $("cameraFilter").value || null,
    has_location: $("locationToggle").checked ? true : null,
  };
}

function clearFilters() {
  $("searchInput").value = "";
  $("dateFrom").value = "";
  $("dateTo").value = "";
  $("cameraFilter").value = "";
  $("locationToggle").checked = false;
  $("sortSelect").value = "relevance";
  state.selectedPeople = [];
  renderSelectedPeople();
  state.similarTo = null;
  $("similarBanner").classList.add("hidden");
  search(1);
}

async function search(page = 1, { keepPanel = false } = {}) {
  clearError();
  if (!keepPanel) togglePeoplePanel(false);
  state.page = page;
  state.similarTo = null;
  $("similarBanner").classList.add("hidden");
  $("photoGrid").innerHTML = skeletonGrid();
  const query = $("searchInput").value.trim();
  startLoad();
  try {
    const result = await request("/search", {
      method: "POST",
      body: JSON.stringify({
        query: query || null,
        ...currentFilters(),
        sort: $("sortSelect").value,
        page,
        per_page: state.perPage,
      }),
    });
    state.total = result.total;
    rememberSearch(query);
    renderPhotos(result);
    const took = result.took_ms ? ` · ${Math.round(result.took_ms)} MS` : "";
    $("resultCount").textContent =
      `${result.total.toLocaleString()} ${result.total === 1 ? "PHOTO" : "PHOTOS"}${took}`;
  } catch (error) {
    $("photoGrid").innerHTML = "";
    showError(`Search failed: ${error.message}`);
  } finally {
    endLoad();
  }
}

async function showSimilar(imageId, filename) {
  clearError();
  setView("photos");
  $("photoGrid").innerHTML = skeletonGrid();
  startLoad();
  try {
    const body = await request(`/images/${imageId}/similar?limit=48`);
    state.similarTo = { image_id: imageId, filename };
    $("similarText").textContent = `Photos that look like ${filename || "the selected photo"}.`;
    $("similarBanner").classList.remove("hidden");
    renderPhotos({ results: body.results, total: body.results.length, page: 1, per_page: state.perPage });
    $("resultCount").textContent = `${body.results.length.toLocaleString()} SIMILAR`;
    $("pager").classList.add("hidden");
  } catch (error) {
    $("photoGrid").innerHTML = "";
    showError(`Similar search failed: ${error.message}`);
  } finally {
    endLoad();
  }
}

function browsePerson(personId) {
  setView("photos");
  closePerson();
  closePhoto();
  $("searchInput").value = "";
  state.selectedPeople = [Number(personId)];
  renderSelectedPeople();
  $("sortSelect").value = "date_desc";
  search(1);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderPhotos(result) {
  const grid = $("photoGrid");
  if (!result.results.length) {
    grid.innerHTML = '<div class="empty">No photos matched. Broaden the description, widen the dates, or remove a person filter.</div>';
  } else {
    const offset = (result.page - 1) * result.per_page;
    grid.innerHTML = result.results.map((photo, i) => `
      <button class="photo" data-image-id="${photo.image_id}" data-filename="${escapeHtml(photo.filename || "")}" aria-label="Open ${escapeHtml(photo.filename || "photo")}">
        <img loading="lazy" src="${API}/images/${photo.image_id}/thumbnail?size=grid&format=webp" alt="${escapeHtml(photo.filename || "Photo")}">
        <span class="frame-no mono">${String(offset + i + 1).padStart(3, "0")}</span>
        <span class="photo-meta mono">
          <span>${escapeHtml(formatDate(photo.taken_at))}</span>
          <span>${photo.face_count ? `${photo.face_count}👤` : ""}</span>
        </span>
        ${typeof photo.score === "number" ? `<span class="score mono" title="Match strength">${Math.round(photo.score * 100)}</span>` : ""}
      </button>
    `).join("");
    grid.querySelectorAll(".photo").forEach((button) => {
      button.addEventListener("click", () =>
        openPhoto(Number(button.dataset.imageId), button.dataset.filename));
    });
  }
  const pages = Math.max(1, Math.ceil(result.total / result.per_page));
  $("pageLabel").textContent = `PAGE ${result.page} / ${pages}`;
  $("prevPage").disabled = result.page <= 1;
  $("nextPage").disabled = result.page >= pages;
  $("pager").classList.toggle("hidden", result.total <= result.per_page);
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

async function loadTimeline() {
  try {
    const body = await request("/timeline");
    state.months = body.months || [];
    renderTimeline();
  } catch {
    state.months = [];
  }
}

function renderTimeline() {
  const box = $("timeline");
  const months = state.months;
  if (months.length < 2) {
    box.classList.add("hidden");
    return;
  }
  const max = Math.max(...months.map((m) => m.count));
  box.classList.remove("hidden");
  box.innerHTML = months.map((m, i) => {
    const height = Math.max(8, Math.round((m.count / max) * 100));
    const [year, month] = m.month.split("-");
    const label = month === "01" || i === 0
      ? `<span class="tl-year mono">${year}</span>` : "";
    return `<button class="tl-bar" type="button" data-month="${m.month}" title="${m.month} · ${m.count} photos">
      <span style="height:${height}%"></span>${label}</button>`;
  }).join("");
  box.querySelectorAll(".tl-bar").forEach((bar) => {
    bar.addEventListener("click", () => {
      const [year, month] = bar.dataset.month.split("-").map(Number);
      const last = new Date(year, month, 0).getDate();
      $("dateFrom").value = `${bar.dataset.month}-01`;
      $("dateTo").value = `${bar.dataset.month}-${String(last).padStart(2, "0")}`;
      search(1);
    });
  });
}

// ---------------------------------------------------------------------------
// Photo modal
// ---------------------------------------------------------------------------

async function openPhoto(imageId, filename = "") {
  $("photoModal").classList.remove("hidden");
  $("modalImage").src = `${API}/images/${imageId}`;
  $("modalName").textContent = filename || "Loading…";
  $("modalMeta").textContent = "";
  $("modalExif").classList.add("hidden");
  $("modalExif").innerHTML = "";
  $("modalFaces").classList.add("hidden");
  $("modalFacesRow").innerHTML = "";
  $("faceActions").classList.add("hidden");
  state.selectedModalFace = null;
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
      details.width && details.height ? `${details.width}×${details.height}` : "",
    ].filter(Boolean).join(" · ").toUpperCase();
    renderExif(details);
    renderModalFaces(details);
  } catch {
    $("modalName").textContent = filename || "Photo";
  }
}

function renderExif(details) {
  const rows = [];
  const add = (key, value) => { if (value) rows.push([key, value]); };
  add("taken", details.taken_at ? new Date(details.taken_at).toLocaleString() : "");
  add("camera", details.camera);
  add("size", details.width && details.height ? `${details.width} × ${details.height}` : "");
  add("file", details.file_size ? formatBytes(details.file_size) : "");
  add("folder", details.folder);
  if (details.lat != null && details.lon != null) {
    add("location", `${details.lat.toFixed(5)}, ${details.lon.toFixed(5)}`);
  }
  add("place", details.place);
  $("modalDetailsBtn").classList.toggle("hidden", !rows.length);
  $("modalExif").innerHTML = rows.map(([k, v]) =>
    `<div class="exif-row"><span class="exif-key mono">${escapeHtml(k.toUpperCase())}</span><span class="exif-val">${escapeHtml(v)}</span></div>`
  ).join("");
}

function renderModalFaces(details) {
  const faces = details.faces || [];
  if (!faces.length) return;
  const names = new Map((details.people || []).map((p) => [p.person_id, p.name]));
  $("modalFaces").classList.remove("hidden");
  $("modalFacesRow").innerHTML = faces.map((f) => {
    const known = f.person_id >= 0;
    const person = known ? personById(f.person_id) : null;
    const label = known
      ? (names.get(f.person_id) || person?.name || `Person ${f.person_id}`)
      : "Unknown";
    return `<div class="modal-face">
      <button class="face-tile ${known ? "" : "unknown"}" type="button" data-face-id="${f.face_id}"
        data-person-id="${f.person_id}" ${known ? "" : "disabled"} aria-label="${escapeHtml(label)}">
        <img class="face-img" loading="lazy" src="${faceCropUrl(f.face_id)}" alt="">
      </button>
      <span class="modal-face-name">${escapeHtml(label)}</span>
    </div>`;
  }).join("");
  $("modalFacesRow").querySelectorAll(".face-tile:not(.unknown)").forEach((tile) => {
    tile.addEventListener("click", () => {
      const personId = Number(tile.dataset.personId);
      const already = state.selectedModalFace === personId;
      state.selectedModalFace = already ? null : personId;
      $("modalFacesRow").querySelectorAll(".face-tile").forEach((t) =>
        t.classList.toggle("selected", Number(t.dataset.personId) === state.selectedModalFace));
      const actions = $("faceActions");
      if (state.selectedModalFace == null) {
        actions.classList.add("hidden");
        return;
      }
      const person = personById(personId);
      $("faceActionsLabel").textContent =
        (person ? personLabel(person) : `Person ${personId}`).toUpperCase();
      actions.classList.remove("hidden");
      $("faceProfile").onclick = () => {
        closePhoto();
        openPerson(personId);
      };
      $("faceSearch").onclick = () => browsePerson(personId);
    });
  });
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
  $("peopleGrid").innerHTML = skeletonPeople();
  startLoad();
  try {
    await loadPeople();
    renderPeopleGrid();
    loadMergeSuggestions();
  } finally {
    endLoad();
  }
}

function renderPeopleGrid() {
  const grid = $("peopleGrid");
  if (!state.people.length) {
    grid.innerHTML = '<div class="empty">No people yet. Add a folder with photos of people, then check back.</div>';
    return;
  }
  grid.innerHTML = state.people.map((person) => `
    <button class="person-tile" type="button" data-person-id="${person.person_id}">
      ${coverHtml(person, "lg")}
      <span class="person-tile-name">${escapeHtml(personLabel(person))}</span>
      <span class="person-tile-count mono">${person.photo_count} ${person.photo_count === 1 ? "PHOTO" : "PHOTOS"}</span>
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
      <div class="merge-card">
        <div class="merge-faces">
          ${coverHtml(s.source)}
          <span class="merge-arrow" aria-hidden="true">→</span>
          ${coverHtml(s.target)}
        </div>
        <div class="merge-copy">
          <strong>${escapeHtml(personLabel(s.source))}</strong> looks like
          <strong>${escapeHtml(personLabel(s.target))}</strong>
          <span class="merge-sim mono">${Math.round(s.similarity * 100)}% SIMILAR</span>
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
  renderSelectedPeople();
  loadMergeSuggestions();
}

// ---------------------------------------------------------------------------
// Person modal
// ---------------------------------------------------------------------------

async function openPerson(personId) {
  let person = personById(personId);
  if (!person) {
    try {
      person = await request(`/people/${personId}`);
    } catch (error) {
      showError(`Could not open the person: ${error.message}`);
      return;
    }
  }
  state.currentPerson = person;
  state.selectedFaces = new Set();
  state.forgetArmed = false;
  state.mergeArmedId = null;

  $("personModal").classList.remove("hidden");
  $("personName").value = person.name || "";
  $("personCover").innerHTML = coverHtml(person, "xl");
  $("personMeta").textContent =
    `${person.photo_count} ${person.photo_count === 1 ? "PHOTO" : "PHOTOS"} · ${person.face_count} ${person.face_count === 1 ? "FACE" : "FACES"}`;
  $("personHide").textContent = person.hidden ? "Unhide" : "Hide";
  $("personForget").textContent = "Forget person";
  $("mergePicker").classList.add("hidden");
  $("mergeFilter").value = "";
  $("detachSelected").disabled = true;
  $("detachSelected").textContent = "Not this person";
  $("personFaces").innerHTML = '<div class="empty slim-empty">Loading faces…</div>';
  $("personSuggestSection").classList.add("hidden");

  loadPersonFaces(person.person_id);
  loadPersonSuggestions(person.person_id);
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
          `${person.photo_count} ${person.photo_count === 1 ? "PHOTO" : "PHOTOS"} · ${person.face_count} ${person.face_count === 1 ? "FACE" : "FACES"}`;
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
    // Suggestions are decoration; the panel simply stays hidden on failure.
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
    renderSelectedPeople();
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
          <span class="merge-item-count mono">${p.photo_count} PHOTOS</span>
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
        openPerson(targetId);
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
    renderSelectedPeople();
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
// Manage view: folders, duplicates, models
// ---------------------------------------------------------------------------

async function loadManageView() {
  loadRoots();
  refreshModels().catch(() => {});
}

async function loadRoots() {
  try {
    const body = await request("/admin/roots");
    state.roots = body.roots || [];
  } catch {
    state.roots = [];
  }
  renderRoots();
}

function renderRoots() {
  const list = $("rootsList");
  if (!state.roots.length) {
    list.innerHTML = '<div class="empty slim-empty">No folders yet. Add the folder where your photos live — scattered drives and project folders all welcome.</div>';
    $("rescanAllButton").disabled = true;
    return;
  }
  $("rescanAllButton").disabled = Boolean(state.activeJob);
  list.innerHTML = state.roots.map((root, i) => `
    <div class="root-row ${root.exists ? "" : "missing"}">
      <div class="root-info">
        <span class="root-path mono">${escapeHtml(root.path)}</span>
        <span class="root-meta mono">${root.exists
          ? `${root.photo_count.toLocaleString()} PHOTOS INDEXED`
          : "FOLDER NOT FOUND — drive unplugged?"}</span>
      </div>
      <div class="root-actions">
        <button class="btn slim-btn" type="button" data-rescan="${i}" ${root.exists ? "" : "disabled"}>Rescan</button>
        <button class="btn ghost slim-btn" type="button" data-forget="${i}">Forget</button>
      </div>
    </div>
  `).join("");
  list.querySelectorAll("[data-rescan]").forEach((button) => {
    button.addEventListener("click", async () => {
      const root = state.roots[Number(button.dataset.rescan)];
      await startIndex(root.path);
      loadRoots();
    });
  });
  list.querySelectorAll("[data-forget]").forEach((button) => {
    button.addEventListener("click", async () => {
      const root = state.roots[Number(button.dataset.forget)];
      try {
        const body = await request(
          `/admin/roots?path=${encodeURIComponent(root.path)}`, { method: "DELETE" });
        state.roots = body.roots || [];
        renderRoots();
      } catch (error) {
        showError(`Could not forget the folder: ${error.message}`);
      }
    });
  });
}

async function addRootFlow() {
  const path = await chooseFolder(false);
  if (!path) return;
  try {
    const body = await request("/admin/roots", {
      method: "POST",
      body: JSON.stringify({ folder: path }),
    });
    state.roots = body.roots || [];
    renderRoots();
    await startIndex(path);
    loadRoots();
  } catch (error) {
    showError(`Could not add the folder: ${error.message}`);
  }
}

async function rescanAll() {
  const targets = state.roots.filter((r) => r.exists);
  if (!targets.length) return;
  $("rescanAllButton").disabled = true;
  try {
    for (const root of targets) {
      const job = await startIndex(root.path);
      if (!job || job.status === "failed") break;
    }
  } finally {
    $("rescanAllButton").disabled = false;
    loadRoots();
  }
}

async function loadDupes() {
  const list = $("dupeList");
  $("findDupes").disabled = true;
  list.innerHTML = '<div class="empty slim-empty">Comparing every photo…</div>';
  startLoad();
  try {
    const body = await request("/duplicates");
    const groups = body || [];
    if (!groups.length) {
      list.innerHTML = '<div class="empty slim-empty">No duplicates found — your library is tidy.</div>';
      return;
    }
    list.innerHTML = groups.map((group) => {
      const shown = group.image_ids.slice(0, 8);
      const extra = group.image_ids.length - shown.length;
      return `<div class="dupe-group">
        <span class="dupe-kind mono ${group.kind === "identical" ? "hard" : ""}">${group.kind.toUpperCase()}</span>
        <div class="dupe-thumbs">
          ${shown.map((id) => `<button class="dupe-thumb" type="button" data-image-id="${id}">
            <img loading="lazy" src="${API}/images/${id}/thumbnail?size=grid&format=webp" alt=""></button>`).join("")}
          ${extra > 0 ? `<span class="dupe-more mono">+${extra}</span>` : ""}
        </div>
      </div>`;
    }).join("");
    list.querySelectorAll(".dupe-thumb").forEach((thumb) => {
      thumb.addEventListener("click", () => openPhoto(Number(thumb.dataset.imageId)));
    });
  } catch (error) {
    list.innerHTML = "";
    showError(`Duplicate scan failed: ${error.message}`);
  } finally {
    $("findDupes").disabled = false;
    endLoad();
  }
}

// ---------------------------------------------------------------------------
// Stats and startup
// ---------------------------------------------------------------------------

async function loadStats() {
  const stats = await request("/stats");
  $("topStats").textContent = [
    `${(stats.total_images || 0).toLocaleString()} PHOTOS`,
    `${(stats.total_people || 0).toLocaleString()} PEOPLE`,
    `${(stats.total_faces || 0).toLocaleString()} FACES`,
  ].join(" · ");
}

async function loadCameras() {
  try {
    const body = await request("/cameras");
    const cameras = body.cameras || [];
    const select = $("cameraFilter");
    if (!cameras.length) {
      select.classList.add("hidden");
      return;
    }
    const current = select.value;
    select.classList.remove("hidden");
    select.innerHTML = '<option value="">Any camera</option>' + cameras.map((c) =>
      `<option value="${escapeHtml(c.camera)}">${escapeHtml(c.camera)} (${c.count})</option>`
    ).join("");
    if ([...select.options].some((o) => o.value === current)) select.value = current;
  } catch {
    $("cameraFilter").classList.add("hidden");
  }
}

async function loadLibrary() {
  await Promise.all([loadPeople(), loadStats(), loadCameras(), loadTimeline()]);
  renderRecentSearches();
  renderSelectedPeople();
  await search(1);
  if (state.view === "people") loadPeopleView();
  if (state.view === "manage") loadManageView();
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
  $("browseButton").addEventListener("click", () => chooseFolder(true));
  $("indexButton").addEventListener("click", startIndexFromSetup);
  $("modelButton").addEventListener("click", fetchModels);
  $("cancelJob").addEventListener("click", cancelJob);
  $("dismissError").addEventListener("click", clearError);

  $("tabPhotos").addEventListener("click", () => setView("photos"));
  $("tabPeople").addEventListener("click", () => setView("people"));
  $("tabManage").addEventListener("click", () => setView("manage"));

  $("searchForm").addEventListener("submit", (event) => {
    event.preventDefault();
    search(1);
  });
  $("sortSelect").addEventListener("change", () => search(1));
  $("dateFrom").addEventListener("change", () => search(1));
  $("dateTo").addEventListener("change", () => search(1));
  $("cameraFilter").addEventListener("change", () => search(1));
  $("locationToggle").addEventListener("change", () => search(1));
  $("clearFilters").addEventListener("click", clearFilters);
  $("similarClear").addEventListener("click", () => {
    state.similarTo = null;
    $("similarBanner").classList.add("hidden");
    search(1);
  });
  $("prevPage").addEventListener("click", () => search(state.page - 1));
  $("nextPage").addEventListener("click", () => search(state.page + 1));

  $("peopleButton").addEventListener("click", () => togglePeoplePanel());
  $("peopleSearch").addEventListener("input", () => renderPeopleOptions($("peopleSearch").value));
  $("peopleModeRow").querySelectorAll("input[name=peopleMode]").forEach((radio) => {
    radio.addEventListener("change", () => {
      state.peopleMode = radio.value;
      renderSelectedPeople();
      if (state.selectedPeople.length > 1) search(1, { keepPanel: true });
    });
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".people-picker")) togglePeoplePanel(false);
  });

  $("modalClose").addEventListener("click", closePhoto);
  $("photoModal").addEventListener("click", (event) => {
    if (event.target === $("photoModal")) closePhoto();
  });
  $("modalDetailsBtn").addEventListener("click", () =>
    $("modalExif").classList.toggle("hidden"));

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

  $("addRootButton").addEventListener("click", addRootFlow);
  $("rescanAllButton").addEventListener("click", rescanAll);
  $("findDupes").addEventListener("click", loadDupes);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      togglePeoplePanel(false);
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
