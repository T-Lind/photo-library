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
  untaggedOnly: false,    // photos with faces but no identified people
  near: null,             // {lat, lon, km} — "photos taken near here"
  results: [],            // photos currently in the grid, for modal navigation
  modalIndex: -1,         // position of the open photo within results
  similarTo: null,
  currentPerson: null,
  selectedFaces: new Set(),
  selectedModalFace: null,
  modalImageId: null,
  forgetArmed: false,
  mergeArmedId: null,
  albums: [],
  currentAlbum: null,
  albumDeleteArmed: false,
  loadCount: 0,
  selection: new Set(),   // image_ids picked for a batch action
  selectionAnchor: -1,    // index of the last toggle, for shift-ranges
  selectionScope: "photos", // "photos" or "album" — where the picks live
  trashArmed: false,
  imageQuery: null,       // {url, label, blob} — active search-by-image chip
  modalTrashArmed: false, // Delete pressed once in the viewer, awaiting confirm
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

function formatDuration(ms) {
  const total = Math.max(0, Math.round((ms || 0) / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h
    ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`;
}

function videoBadge(photo) {
  return photo.media_type === "video"
    ? `<span class="video-badge mono">▶ ${formatDuration(photo.duration_ms)}</span>`
    : "";
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
  for (const [id, name] of [["photosView", "photos"], ["peopleView", "people"],
                            ["albumsView", "albums"], ["manageView", "manage"]]) {
    $(id).classList.toggle("hidden", view !== name);
  }
  for (const [id, name] of [["tabPhotos", "photos"], ["tabPeople", "people"],
                            ["tabAlbums", "albums"], ["tabManage", "manage"]]) {
    $(id).classList.toggle("active", view === name);
    $(id).setAttribute("aria-selected", String(view === name));
  }
  if (view === "people") loadPeopleView();
  if (view === "albums") loadAlbumsView();
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

function setImageQuery(next) {
  if (state.imageQuery?.blob) URL.revokeObjectURL(state.imageQuery.url);
  state.imageQuery = next;
}

function clearImageQuery() {
  if (!state.imageQuery) return;
  setImageQuery(null);
  renderSelectedPeople();
}

function renderSelectedPeople() {
  const box = $("selectedPeople");
  const count = state.selectedPeople.length;
  $("peopleButton").textContent = count ? `People (${count})` : "People";
  $("peopleButton").classList.toggle("active-filter", count > 0);
  const parts = [];
  if (state.imageQuery) {
    parts.push(`<button class="chip person-chip image-chip" type="button" data-image-query="1" title="Stop searching by this image">
      <img src="${state.imageQuery.url}" alt=""><span>${escapeHtml(state.imageQuery.label)}</span> ×</button>`);
  }
  if (count > 1) {
    parts.push(`<span class="chips-label mono">${state.peopleMode === "all" ? "ALL OF" : "ANY OF"}</span>`);
  }
  parts.push(...state.selectedPeople.map((id) => {
    const p = personById(id);
    return `<button class="chip person-chip" type="button" data-person-id="${id}" title="Remove from search">
      ${escapeHtml(p ? personLabel(p) : `Person ${id}`)} ×</button>`;
  }));
  if (state.untaggedOnly) {
    parts.push('<button class="chip person-chip" type="button" data-untagged="1" title="Remove filter">no one tagged ×</button>');
  }
  if (state.near) {
    parts.push(`<button class="chip person-chip" type="button" data-near="1" title="Remove filter">near ${state.near.lat.toFixed(4)}, ${state.near.lon.toFixed(4)} ×</button>`);
  }
  if (!parts.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  box.innerHTML = parts.join("");
  box.querySelectorAll(".person-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      if (chip.dataset.imageQuery) {
        setImageQuery(null);
        state.similarTo = null;
        $("similarBanner").classList.add("hidden");
      } else if (chip.dataset.untagged) state.untaggedOnly = false;
      else if (chip.dataset.near) state.near = null;
      else {
        state.selectedPeople = state.selectedPeople.filter(
          (id) => id !== Number(chip.dataset.personId));
      }
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
    media: $("mediaFilter").value || null,
    has_location: $("locationToggle").checked ? true : null,
    untagged_only: state.untaggedOnly,
    near_lat: state.near ? state.near.lat : null,
    near_lon: state.near ? state.near.lon : null,
    near_km: state.near ? state.near.km : 1.0,
  };
}

function clearFilters() {
  $("searchInput").value = "";
  $("dateFrom").value = "";
  $("dateTo").value = "";
  $("cameraFilter").value = "";
  $("mediaFilter").value = "";
  $("locationToggle").checked = false;
  $("sortSelect").value = "relevance";
  state.selectedPeople = [];
  state.untaggedOnly = false;
  state.near = null;
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
  clearImageQuery();
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
    setImageQuery({
      url: `${API}/images/${imageId}/thumbnail?size=grid&format=webp`,
      label: `like ${filename || "this photo"}`,
    });
    renderSelectedPeople();
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
  state.results = result.results;
  if (!result.results.length) {
    grid.innerHTML = '<div class="empty">No photos matched. Broaden the description, widen the dates, or remove a person filter.</div>';
  } else {
    const offset = (result.page - 1) * result.per_page;
    grid.innerHTML = result.results.map((photo, i) => `
      <button class="photo ${state.selection.has(photo.image_id) ? "selected" : ""}" data-image-id="${photo.image_id}" data-index="${i}" data-filename="${escapeHtml(photo.filename || "")}" aria-label="Open ${escapeHtml(photo.filename || "photo")}">
        <img loading="lazy" src="${API}/images/${photo.image_id}/thumbnail?size=grid&format=webp" alt="${escapeHtml(photo.filename || "Photo")}">
        <span class="select-check" title="Select (Ctrl-click works too, Shift-click for a range)">✓</span>
        <span class="frame-no mono">${String(offset + i + 1).padStart(3, "0")}</span>
        ${videoBadge(photo)}
        <span class="photo-meta mono">
          <span>${escapeHtml(formatDate(photo.taken_at))}</span>
          <span>${photo.face_count ? `${photo.face_count}👤` : ""}</span>
        </span>
        ${photo.text_match ? '<span class="score mono text-hit" title="The query matched text inside this image">TEXT</span>'
          : typeof photo.score === "number" ? `<span class="score mono" title="Match strength">${Math.round(photo.score * 100)}</span>` : ""}
      </button>
    `).join("");
    grid.querySelectorAll(".photo").forEach((button) => {
      button.addEventListener("click", (event) => {
        if (handleSelectClick(event, button, "photos",
            (i) => state.results[i]?.image_id)) return;
        openPhoto(Number(button.dataset.imageId), button.dataset.filename);
      });
    });
  }
  const pages = Math.max(1, Math.ceil(result.total / result.per_page));
  $("pageLabel").textContent = `PAGE ${result.page} / ${pages}`;
  $("prevPage").disabled = result.page <= 1;
  $("nextPage").disabled = result.page >= pages;
  $("pager").classList.toggle("hidden", result.total <= result.per_page);
}

// ---------------------------------------------------------------------------
// Multi-select + batch actions
// ---------------------------------------------------------------------------
// The selection is a set of image_ids, so it survives page changes; the
// anchor index only powers Shift-ranges within the currently visible grid.

function handleSelectClick(event, tile, scope, idAt) {
  const imageId = Number(tile.dataset.imageId);
  const index = Number(tile.dataset.index);
  if (event.shiftKey && state.selection.size &&
      state.selectionScope === scope && state.selectionAnchor >= 0) {
    for (let i = Math.min(state.selectionAnchor, index);
         i <= Math.max(state.selectionAnchor, index); i++) {
      const id = idAt(i);
      if (id != null) state.selection.add(id);
    }
    state.selectionAnchor = index;
    reflectSelection();
    return true;
  }
  if (event.ctrlKey || event.metaKey || event.target.closest(".select-check")) {
    if (state.selectionScope !== scope && state.selection.size) {
      state.selection.clear();  // picks from two different grids never mix
    }
    state.selectionScope = scope;
    if (state.selection.has(imageId)) state.selection.delete(imageId);
    else state.selection.add(imageId);
    state.selectionAnchor = index;
    reflectSelection();
    return true;
  }
  return false;
}

function reflectSelection() {
  state.trashArmed = false;
  document.querySelectorAll(".photo[data-image-id]").forEach((tile) => {
    tile.classList.toggle("selected",
      state.selection.has(Number(tile.dataset.imageId)));
  });
  const n = state.selection.size;
  $("selectionBar").classList.toggle("hidden", n === 0);
  $("selectionCount").textContent =
    `${n} SELECTED${state.selectionScope === "album" ? " · IN ALBUM" : ""}`;
  $("selTrash").textContent = n === 1 ? "Trash" : `Trash ${n}`;
  $("selRemoveAlbum").classList.toggle("hidden",
    !(state.selectionScope === "album" && state.currentAlbum));
  if (n === 0) $("selAlbumPanel").classList.add("hidden");
}

function clearSelection() {
  state.selection.clear();
  state.selectionAnchor = -1;
  reflectSelection();
}

function toggleSelectionAlbumPanel(show) {
  const panel = $("selAlbumPanel");
  const wanted = show ?? panel.classList.contains("hidden");
  panel.classList.toggle("hidden", !wanted);
  $("selAlbumBtn").setAttribute("aria-expanded", String(wanted));
  if (!wanted) return;
  const list = $("selAlbumList");
  list.innerHTML = state.albums.length
    ? state.albums.map((album) => `
        <button class="picker-row" type="button" data-album-id="${album.album_id}">
          <span class="picker-name">${escapeHtml(album.name)}</span>
          <span class="picker-count mono">${album.photo_count}</span>
        </button>`).join("")
    : '<div class="empty slim-empty">No albums yet — name one below.</div>';
  list.querySelectorAll(".picker-row").forEach((row) => {
    row.addEventListener("click", () =>
      batchAddToAlbum(Number(row.dataset.albumId)));
  });
}

async function batchAddToAlbum(albumId) {
  const ids = [...state.selection];
  if (!ids.length) return;
  try {
    const body = await request(`/albums/${albumId}/items`, {
      method: "POST",
      body: JSON.stringify({ image_ids: ids }),
    });
    await loadAlbums();
    toggleSelectionAlbumPanel(false);
    $("selAlbumBtn").textContent = `Added ${body.added} ✓`;
    setTimeout(() => { $("selAlbumBtn").textContent = "Add to album"; }, 1600);
    clearSelection();
    if (state.currentAlbum?.album_id === albumId
        && !$("albumDetail").classList.contains("hidden")) {
      openAlbum(albumId);
    }
  } catch (error) {
    showError(`Could not add the photos: ${error.message}`);
  }
}

async function batchRemoveFromAlbum() {
  const album = state.currentAlbum;
  const ids = [...state.selection];
  if (!album || !ids.length) return;
  try {
    await request(`/albums/${album.album_id}/items/remove`, {
      method: "POST",
      body: JSON.stringify({ image_ids: ids }),
    });
    clearSelection();
    openAlbum(album.album_id);
  } catch (error) {
    showError(`Could not remove the photos: ${error.message}`);
  }
}

async function batchTrash() {
  const ids = [...state.selection];
  if (!ids.length) return;
  const button = $("selTrash");
  if (!state.trashArmed) {
    state.trashArmed = true;
    button.textContent = `Really trash ${ids.length}? → Recycle Bin`;
    return;
  }
  button.disabled = true;
  startLoad();
  try {
    const body = await request("/images/trash", {
      method: "POST",
      body: JSON.stringify({ image_ids: ids }),
    });
    if (body.failed?.length) {
      showError(`${body.failed.length} photo(s) could not be trashed — they stay in the library.`);
    }
    const fromAlbum = state.selectionScope === "album" && state.currentAlbum;
    clearSelection();
    await Promise.all([loadStats(), loadPeople(), loadTimeline()]);
    if (fromAlbum) openAlbum(state.currentAlbum.album_id);
    else if (state.view === "photos") search(state.page);
  } catch (error) {
    showError(`Trash failed: ${error.message}`);
  } finally {
    button.disabled = false;
    endLoad();
  }
}

async function batchExport() {
  const ids = [...state.selection];
  if (!ids.length) return;
  const button = $("selExport");
  const folder = await chooseFolder(false);
  if (!folder) return;
  button.disabled = true;
  startLoad();
  try {
    const body = await request("/images/export", {
      method: "POST",
      body: JSON.stringify({ image_ids: ids, folder }),
    });
    button.textContent = `Copied ${body.copied} ✓`;
    if (body.missing || body.failed?.length) {
      showError(`${body.copied} copied — but ${body.missing || 0} original(s) are missing` +
        `${body.failed?.length ? ` and ${body.failed.length} failed` : ""}.`);
    }
    setTimeout(() => { button.textContent = "Export copies…"; }, 2200);
  } catch (error) {
    showError(`Export failed: ${error.message}`);
  } finally {
    button.disabled = false;
    endLoad();
  }
}

function selectAllVisible() {
  const albumOpen = state.view === "albums"
    && !$("albumDetail").classList.contains("hidden");
  const scope = albumOpen ? "album" : "photos";
  const ids = albumOpen
    ? (state.currentAlbum?.images || []).map((r) => r.image_id)
    : state.results.map((r) => r.image_id);
  if (!ids.length) return;
  if (state.selectionScope !== scope) state.selection.clear();
  state.selectionScope = scope;
  ids.forEach((id) => state.selection.add(id));
  state.selectionAnchor = 0;
  reflectSelection();
}

// ---------------------------------------------------------------------------
// Keyboard cheat sheet (?)
// ---------------------------------------------------------------------------

function toggleShortcuts(show) {
  const modal = $("shortcutModal");
  const wanted = show ?? modal.classList.contains("hidden");
  modal.classList.toggle("hidden", !wanted);
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

// The stage shows either the <img> (with zoom/pan) or the <video>.
function modalVideoActive() {
  return !$("modalVideo").classList.contains("hidden");
}

function stopModalVideo() {
  const video = $("modalVideo");
  if (video.getAttribute("src")) {
    video.pause();
    delete video.dataset.imageId;   // cleared first: the load() below fires
    video.removeAttribute("src");   // an error event we must ignore
    video.removeAttribute("poster");
    video.load();
  }
  video.classList.add("hidden");
  $("videoFallback").classList.add("hidden");
}

function showModalMedia(imageId, isVideo) {
  const img = $("modalImage");
  const video = $("modalVideo");
  $("modalCopyBtn").classList.toggle("hidden", Boolean(isVideo));
  if (isVideo) {
    if (img.getAttribute("src")) img.removeAttribute("src");
    delete img.dataset.imageId;
    img.classList.add("hidden");
    resetZoom();
    if (video.dataset.imageId !== String(imageId)) {
      $("videoFallback").classList.add("hidden");
      video.dataset.imageId = String(imageId);
      video.poster = `${API}/images/${imageId}/thumbnail?size=preview&format=jpeg`;
      video.src = `${API}/images/${imageId}`;
    }
    video.classList.remove("hidden");
  } else {
    stopModalVideo();
    img.classList.remove("hidden");
    if (img.dataset.imageId !== String(imageId)) {
      img.dataset.imageId = String(imageId);
      img.src = `${API}/images/${imageId}`;
    }
  }
}

function disarmModalTrash() {
  if (!state.modalTrashArmed) return;
  state.modalTrashArmed = false;
  const meta = $("modalMeta");
  meta.classList.remove("danger-text");
  if (meta.dataset.prev != null) {
    meta.textContent = meta.dataset.prev;
    delete meta.dataset.prev;
  }
}

async function openPhoto(imageId, filename = "") {
  state.modalIndex = state.results.findIndex((r) => r.image_id === Number(imageId));
  state.modalImageId = Number(imageId);
  disarmModalTrash();
  updateModalNav();
  $("albumPickPanel").classList.add("hidden");
  $("modalAlbumBtn").textContent = "Add to album";
  $("modalCopyBtn").textContent = "Copy image";
  $("modalRevealBtn").textContent = "Show in Explorer";
  $("photoModal").classList.remove("hidden");
  resetZoom();
  // The grid row usually knows the media type; details confirm it below.
  const known = (state.modalIndex >= 0 ? state.results[state.modalIndex] : null)
    || (state.currentAlbum?.images || []).find((r) => r.image_id === Number(imageId));
  showModalMedia(imageId, known?.media_type === "video");
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
  $("modalCopyBtn").onclick = () => copyModalImage(imageId, $("modalCopyBtn"));
  $("modalRevealBtn").onclick = () => revealImage(imageId, $("modalRevealBtn"));
  try {
    const details = await request(`/images/${imageId}/details`);
    if (state.modalImageId !== Number(imageId)) return; // user moved on
    showModalMedia(imageId, details.media_type === "video");
    $("modalName").textContent = details.filename || "Photo";
    $("modalMeta").textContent = [
      formatDate(details.taken_at),
      details.media_type === "video" ? formatDuration(details.duration_ms) : "",
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
  add("length", details.media_type === "video" ? formatDuration(details.duration_ms) : "");
  add("camera", details.camera);
  add("size", details.width && details.height ? `${details.width} × ${details.height}` : "");
  add("file", details.file_size ? formatBytes(details.file_size) : "");
  add("folder", details.folder);
  add("place", details.place);
  $("modalDetailsBtn").classList.toggle("hidden",
    !rows.length && !details.ocr_text && details.lat == null);
  const hasGps = details.lat != null && details.lon != null;
  const textBlock = details.ocr_text
    ? `<div class="exif-row text-row"><span class="exif-key mono">TEXT</span>
        <span class="exif-val">${escapeHtml(details.ocr_text.replaceAll("\n", " · ").slice(0, 300))}</span>
        <button id="copyTextBtn" class="btn slim-btn copy-text" type="button"
          title="Copy every word found in this photo">Copy text</button></div>`
    : "";
  $("modalExif").innerHTML = rows.map(([k, v]) =>
    `<div class="exif-row"><span class="exif-key mono">${escapeHtml(k.toUpperCase())}</span><span class="exif-val">${escapeHtml(v)}</span></div>`
  ).join("") + textBlock + (hasGps
    ? `<div class="exif-row"><span class="exif-key mono">LOCATION</span>
        <button class="exif-near" type="button" id="exifNear">${details.lat.toFixed(5)}, ${details.lon.toFixed(5)} · photos near here</button></div>`
    : "");
  if (hasGps) {
    $("exifNear").addEventListener("click", () => {
      state.near = { lat: details.lat, lon: details.lon, km: 1.0 };
      closePhoto();
      setView("photos");
      renderSelectedPeople();
      search(1);
    });
  }
  if (details.ocr_text) {
    $("copyTextBtn").addEventListener("click", () =>
      copyImageText(details.image_id, $("copyTextBtn")));
  }
}

async function copyImageText(imageId, button) {
  try {
    const body = await request(`/images/${imageId}/text`);
    const text = body.text || "";
    if (!text) {
      button.textContent = "No text stored";
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard API can be refused; fall back to the selection trick.
      const scratch = document.createElement("textarea");
      scratch.value = text;
      document.body.appendChild(scratch);
      scratch.select();
      document.execCommand("copy");
      scratch.remove();
    }
    button.textContent = "Copied ✓";
    setTimeout(() => { button.textContent = "Copy text"; }, 1500);
  } catch (error) {
    showError(`Could not copy the text: ${error.message}`);
  }
}

async function copyModalImage(imageId, button) {
  // The preview-size JPEG re-encode is used as the source: unlike the
  // original, it is decodable for every format (HEIC and RAW included).
  try {
    if (!navigator.clipboard || !window.ClipboardItem) {
      throw new Error("the clipboard is not available here");
    }
    const img = new Image();
    img.src = `${API}/images/${imageId}/thumbnail?size=preview&format=jpeg`;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext("2d").drawImage(img, 0, 0);
    const blob = await new Promise((resolve, reject) => canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("could not encode the image"))),
      "image/png"));
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    button.textContent = "Copied ✓";
    setTimeout(() => { button.textContent = "Copy image"; }, 1500);
  } catch (error) {
    showError(`Could not copy the image: ${error.message}`);
  }
}

async function revealImage(imageId, button) {
  try {
    await request(`/images/${imageId}/reveal`, { method: "POST" });
    button.textContent = "Opened ✓";
    setTimeout(() => { button.textContent = "Show in Explorer"; }, 1500);
  } catch (error) {
    showError(`Could not show the file: ${error.message}`);
  }
}

async function trashOpenPhoto() {
  const imageId = state.modalImageId;
  if (imageId == null) return;
  const meta = $("modalMeta");
  if (!state.modalTrashArmed) {
    state.modalTrashArmed = true;
    meta.dataset.prev = meta.textContent;
    meta.textContent = "PRESS DELETE AGAIN — MOVES THE FILE TO THE RECYCLE BIN";
    meta.classList.add("danger-text");
    setTimeout(disarmModalTrash, 4000);
    return;
  }
  disarmModalTrash();
  startLoad();
  try {
    const body = await request("/images/trash", {
      method: "POST",
      body: JSON.stringify({ image_ids: [imageId] }),
    });
    if (body.failed?.length) {
      showError("The file could not be moved to the Recycle Bin — it stays in the library.");
      return;
    }
    closePhoto();
    state.selection.delete(imageId);
    reflectSelection();
    await Promise.all([loadStats(), loadPeople(), loadTimeline()]);
    if (state.currentAlbum && !$("albumDetail").classList.contains("hidden")) {
      openAlbum(state.currentAlbum.album_id);
    }
    if (state.view === "photos") search(state.page);
  } catch (error) {
    showError(`Trash failed: ${error.message}`);
  } finally {
    endLoad();
  }
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

// ---------------------------------------------------------------------------
// Viewer zoom + pan
// ---------------------------------------------------------------------------
// A plain CSS transform on the image: translate then scale about the centre.
// For the point under the cursor to stay put when the scale changes from s
// to s', the offset moves to x' = c - (c - x) * s'/s.

const zoom = { scale: 1, x: 0, y: 0, dragging: false, sx: 0, sy: 0 };

function applyZoom() {
  const img = $("modalImage");
  img.style.transform = zoom.scale === 1
    ? "" : `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`;
  img.classList.toggle("zoomed", zoom.scale > 1);
}

function resetZoom() {
  zoom.scale = 1;
  zoom.x = 0;
  zoom.y = 0;
  zoom.dragging = false;
  applyZoom();
}

function zoomTowards(clientX, clientY, factor) {
  const rect = $("modalStage").getBoundingClientRect();
  const cx = clientX - rect.left - rect.width / 2;
  const cy = clientY - rect.top - rect.height / 2;
  const next = Math.min(8, Math.max(1, zoom.scale * factor));
  const applied = next / zoom.scale;
  zoom.x = cx - (cx - zoom.x) * applied;
  zoom.y = cy - (cy - zoom.y) * applied;
  zoom.scale = next;
  if (next === 1) { zoom.x = 0; zoom.y = 0; }
  applyZoom();
}

function initZoom() {
  const stage = $("modalStage");
  const img = $("modalImage");
  stage.addEventListener("wheel", (event) => {
    if (modalVideoActive()) return; // the player owns its own gestures
    event.preventDefault();
    zoomTowards(event.clientX, event.clientY,
      Math.exp(-event.deltaY * 0.0016));
  }, { passive: false });
  stage.addEventListener("dblclick", (event) => {
    if (modalVideoActive()) return; // double-click on a video = fullscreen
    if (zoom.scale > 1) resetZoom();
    else zoomTowards(event.clientX, event.clientY, 2.5);
  });
  img.addEventListener("pointerdown", (event) => {
    if (zoom.scale === 1) return;
    zoom.dragging = true;
    zoom.sx = event.clientX - zoom.x;
    zoom.sy = event.clientY - zoom.y;
    img.classList.add("dragging");
    img.setPointerCapture(event.pointerId);
  });
  img.addEventListener("pointermove", (event) => {
    if (!zoom.dragging) return;
    zoom.x = event.clientX - zoom.sx;
    zoom.y = event.clientY - zoom.sy;
    applyZoom();
  });
  const stopDrag = () => {
    zoom.dragging = false;
    img.classList.remove("dragging");
  };
  img.addEventListener("pointerup", stopDrag);
  img.addEventListener("pointercancel", stopDrag);
}

function closePhoto() {
  $("photoModal").classList.add("hidden");
  $("modalImage").removeAttribute("src");
  delete $("modalImage").dataset.imageId;
  stopModalVideo();
  disarmModalTrash();
  resetZoom();
}

function updateModalNav() {
  const i = state.modalIndex;
  const inResults = i >= 0 && state.results.length > 0;
  const pages = Math.max(1, Math.ceil(state.total / state.perPage));
  const canPage = !state.similarTo;
  const hasPrev = inResults && (i > 0 || (canPage && state.page > 1));
  const hasNext = inResults && (i < state.results.length - 1
    || (canPage && state.page < pages));
  $("modalPrev").classList.toggle("hidden", !hasPrev);
  $("modalNext").classList.toggle("hidden", !hasNext);
}

async function navigatePhoto(delta) {
  const i = state.modalIndex;
  if (i < 0 || !state.results.length) return;
  const next = i + delta;
  if (next >= 0 && next < state.results.length) {
    const photo = state.results[next];
    openPhoto(photo.image_id, photo.filename || "");
    return;
  }
  // Walked off the page — fetch the neighbouring one and keep going.
  if (state.similarTo) return;
  const pages = Math.max(1, Math.ceil(state.total / state.perPage));
  if (delta > 0 && state.page < pages) {
    await search(state.page + 1);
    if (state.results.length) {
      const photo = state.results[0];
      openPhoto(photo.image_id, photo.filename || "");
    }
  } else if (delta < 0 && state.page > 1) {
    await search(state.page - 1);
    if (state.results.length) {
      const photo = state.results[state.results.length - 1];
      openPhoto(photo.image_id, photo.filename || "");
    }
  }
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
    loadUntagged();
  } finally {
    endLoad();
  }
}

async function loadUntagged() {
  try {
    const body = await request("/search", {
      method: "POST",
      body: JSON.stringify({ untagged_only: true, sort: "date_desc", per_page: 12 }),
    });
    const panel = $("untaggedPanel");
    if (!body.total) {
      panel.classList.add("hidden");
      return;
    }
    panel.classList.remove("hidden");
    $("untaggedCopy").textContent =
      `${body.total.toLocaleString()} ${body.total === 1 ? "photo" : "photos"} where a face was spotted but nobody has been identified.`;
    $("untaggedRow").innerHTML = body.results.map((photo) => `
      <button class="dupe-thumb" type="button" data-image-id="${photo.image_id}" data-filename="${escapeHtml(photo.filename || "")}">
        <img loading="lazy" src="${API}/images/${photo.image_id}/thumbnail?size=grid&format=webp" alt="">
      </button>`).join("");
    $("untaggedRow").querySelectorAll(".dupe-thumb").forEach((thumb) => {
      thumb.addEventListener("click", () =>
        openPhoto(Number(thumb.dataset.imageId), thumb.dataset.filename));
    });
  } catch {
    $("untaggedPanel").classList.add("hidden");
  }
}

function renderPeopleGrid() {
  const grid = $("peopleGrid");
  if (!state.people.length) {
    grid.innerHTML = '<div class="empty">No people yet. Add a folder with photos of people, then check back.</div>';
    return;
  }
  grid.innerHTML = state.people.map((person) => `
    <div class="person-tile" tabindex="0" role="button" data-person-id="${person.person_id}"
         aria-label="Open ${escapeHtml(personLabel(person))}">
      ${coverHtml(person, "lg")}
      <span class="person-tile-name" title="Click to rename">${escapeHtml(personLabel(person))}</span>
      <span class="person-tile-count mono">${person.photo_count} ${person.photo_count === 1 ? "PHOTO" : "PHOTOS"}</span>
    </div>
  `).join("");
  grid.querySelectorAll(".person-tile").forEach((tile) => {
    const personId = Number(tile.dataset.personId);
    tile.addEventListener("click", () => openPerson(personId));
    tile.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.target === tile) openPerson(personId);
    });
    tile.querySelector(".person-tile-name").addEventListener("click", (event) => {
      event.stopPropagation();
      startInlineRename(tile, personId);
    });
  });
}

function startInlineRename(tile, personId) {
  const person = personById(personId);
  const span = tile.querySelector(".person-tile-name");
  if (!person || !span || tile.querySelector(".tile-rename")) return;
  const input = document.createElement("input");
  input.className = "field tile-rename";
  input.maxLength = 200;
  input.value = person.name || "";
  input.placeholder = "Name…";
  input.setAttribute("aria-label", "Person name");
  span.replaceWith(input);
  input.focus();
  input.select();
  input.addEventListener("click", (event) => event.stopPropagation());

  let finished = false;
  const finish = async (save) => {
    if (finished) return;
    finished = true;
    const name = input.value.trim();
    if (save && name && name !== person.name) {
      try {
        await request(`/people/${personId}`, {
          method: "PATCH",
          body: JSON.stringify({ name }),
        });
        await loadPeople();
        renderSelectedPeople();
      } catch (error) {
        showError(`Rename failed: ${error.message}`);
      }
    }
    renderPeopleGrid();
  };
  input.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Enter") finish(true);
    if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}

function mergeReviewOpen() {
  return localStorage.getItem("photolib.mergeOpen") === "1";
}

function applyMergeCollapse() {
  const open = mergeReviewOpen();
  $("mergeBody").classList.toggle("hidden", !open);
  $("mergeToggle").setAttribute("aria-expanded", String(open));
  $("mergeToggle").querySelector(".collapse-chevron").textContent = open ? "▾" : "▸";
}

async function loadMergeSuggestions() {
  const list = $("mergeSuggestList");
  applyMergeCollapse();
  try {
    // A low floor keeps the section alive with long-shot candidates; the
    // similarity badge tells the user how seriously to take each one.
    const body = await request("/people/merge-suggestions?limit=12&min_similarity=0.25");
    const dismissed = dismissedMerges();
    const pairs = (body.suggestions || []).filter(
      (s) => !dismissed.has(mergeKey(s.source.person_id, s.target.person_id)));
    $("mergeCount").textContent = pairs.length
      ? `${pairs.length} TO REVIEW` : "ALL CLEAR";
    if (!pairs.length) {
      list.innerHTML = '<div class="empty slim-empty">Nothing left to review — no two people look alike right now.</div>';
      return;
    }
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
    $("mergeCount").textContent = "";
    list.innerHTML = '<div class="empty slim-empty">Merge suggestions are unavailable right now.</div>';
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
      <button class="face-tile" type="button" data-face-id="${f.face_id}" data-image-id="${f.image_id}" title="Quality ${Math.round((f.quality || 0) * 100)}%${f.confirmed ? " · confirmed by you" : ""} — click to select, ↗ opens the photo">
        <img class="face-img" loading="lazy" src="${faceCropUrl(f.face_id)}" alt="">
        ${f.confirmed ? '<span class="face-badge" aria-label="Confirmed">✓</span>' : ""}
        <span class="open-photo mono" data-open="${f.image_id}" title="Open the photo this face is from" role="button">↗</span>
      </button>
    `).join("");
    $("personFaces").querySelectorAll(".open-photo").forEach((corner) => {
      corner.addEventListener("click", (event) => {
        event.stopPropagation();
        openPhoto(Number(corner.dataset.open));
      });
    });
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
  loadOcrStatus();
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
    const body = await request("/admin/duplicates");
    const groups = body.groups || [];
    if (!groups.length) {
      list.innerHTML = '<div class="empty slim-empty">No duplicates found — your library is tidy.</div>';
      return;
    }
    const keeperOf = (group) => {
      const items = group.items
        || group.image_ids.map((id) => ({ image_id: id, file_size: 0 }));
      return items.reduce((a, b) => (b.file_size > a.file_size ? b : a),
                          items[0]).image_id;
    };
    list.innerHTML = groups.map((group, gi) => {
      const items = group.items
        || group.image_ids.map((id) => ({ image_id: id, file_size: 0 }));
      const keeper = keeperOf(group);
      const shown = items.slice(0, 8);
      const extra = items.length - shown.length;
      return `<div class="dupe-group" data-group="${gi}">
        <span class="dupe-kind mono ${group.kind === "identical" ? "hard" : ""}">${group.kind.toUpperCase()}</span>
        <div class="dupe-thumbs">
          ${shown.map((item) => `<button class="dupe-thumb ${item.image_id === keeper ? "keeper" : ""}" type="button" data-image-id="${item.image_id}"
              title="${formatBytes(item.file_size)}${item.image_id === keeper ? " · largest, will be kept" : ""}">
            <img loading="lazy" src="${API}/images/${item.image_id}/thumbnail?size=grid&format=webp" alt=""></button>`).join("")}
          ${extra > 0 ? `<span class="dupe-more mono">+${extra}</span>` : ""}
        </div>
        <button class="btn slim-btn dupe-keep" type="button">Keep largest · trash ${items.length - 1}</button>
      </div>`;
    }).join("");
    list.querySelectorAll(".dupe-thumb").forEach((thumb) => {
      thumb.addEventListener("click", () => openPhoto(Number(thumb.dataset.imageId)));
    });
    list.querySelectorAll(".dupe-keep").forEach((button) => {
      button.addEventListener("click", async () => {
        const group = groups[Number(button.closest(".dupe-group").dataset.group)];
        const losers = group.image_ids.filter((id) => id !== keeperOf(group));
        if (button.dataset.armed !== "1") {
          button.dataset.armed = "1";
          button.textContent = `Really trash ${losers.length}? → Recycle Bin`;
          return;
        }
        button.disabled = true;
        try {
          await request("/images/trash", {
            method: "POST",
            body: JSON.stringify({ image_ids: losers }),
          });
          await Promise.all([loadStats(), loadPeople()]);
          loadDupes();
        } catch (error) {
          showError(`Could not trash the duplicates: ${error.message}`);
          button.disabled = false;
        }
      });
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
// Albums
// ---------------------------------------------------------------------------

async function loadAlbums() {
  try {
    state.albums = (await request("/albums")).albums || [];
  } catch {
    state.albums = [];
  }
}

async function loadAlbumsView() {
  $("albumDetail").classList.add("hidden");
  $("albumsHome").classList.remove("hidden");
  startLoad();
  try {
    await loadAlbums();
    renderAlbumsGrid();
  } finally {
    endLoad();
  }
}

function albumCoverHtml(album) {
  if (album.cover_image_id >= 0) {
    return `<img loading="lazy" src="${API}/images/${album.cover_image_id}/thumbnail?size=grid&format=webp" alt="">`;
  }
  return '<span class="album-blank" aria-hidden="true">▦</span>';
}

function renderAlbumsGrid() {
  const grid = $("albumsGrid");
  if (!state.albums.length) {
    grid.innerHTML = '<div class="empty">No albums yet. Name one above, then add photos from the viewer or the suggestions inside the album.</div>';
    return;
  }
  grid.innerHTML = state.albums.map((album) => `
    <button class="album-card" type="button" data-album-id="${album.album_id}">
      <span class="album-cover">${albumCoverHtml(album)}</span>
      <span class="album-name">${escapeHtml(album.name)}</span>
      <span class="album-count mono">${album.photo_count} ${album.photo_count === 1 ? "PHOTO" : "PHOTOS"}</span>
    </button>
  `).join("");
  grid.querySelectorAll(".album-card").forEach((card) => {
    card.addEventListener("click", () => openAlbum(Number(card.dataset.albumId)));
  });
}

async function createAlbum(name, imageIds = []) {
  const album = await request("/albums", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
  if (imageIds.length) {
    await request(`/albums/${album.album_id}/items`, {
      method: "POST",
      body: JSON.stringify({ image_ids: imageIds }),
    });
  }
  await loadAlbums();
  return album;
}

async function openAlbum(albumId) {
  state.albumDeleteArmed = false;
  $("albumDelete").textContent = "Delete album";
  $("albumsHome").classList.add("hidden");
  $("albumDetail").classList.remove("hidden");
  $("albumPhotos").innerHTML = skeletonGrid(6);
  $("albumSuggestions").innerHTML = "";
  startLoad();
  try {
    const detail = await request(`/albums/${albumId}?limit=500`);
    state.currentAlbum = detail;
    $("albumTitle").value = detail.name;
    $("albumMeta").textContent =
      `${detail.photo_count} ${detail.photo_count === 1 ? "PHOTO" : "PHOTOS"}`;
    renderAlbumPhotos(detail);
    loadAlbumSuggestions(albumId);
  } catch (error) {
    showError(`Could not open the album: ${error.message}`);
    loadAlbumsView();
  } finally {
    endLoad();
  }
}

function renderAlbumPhotos(detail) {
  const grid = $("albumPhotos");
  $("albumPhotosTitle").textContent = `Photos (${detail.images.length})`;
  if (!detail.images.length) {
    grid.innerHTML = '<div class="empty">Empty so far. Add photos from the suggestions above, or from any photo\'s "Add to album" button.</div>';
    return;
  }
  grid.innerHTML = detail.images.map((photo, i) => `
    <div class="photo album-photo ${state.selection.has(photo.image_id) ? "selected" : ""}" data-image-id="${photo.image_id}" data-index="${i}">
      <img loading="lazy" src="${API}/images/${photo.image_id}/thumbnail?size=grid&format=webp" alt="${escapeHtml(photo.filename || "Photo")}">
      <span class="select-check" title="Select (Ctrl-click works too, Shift-click for a range)">✓</span>
      ${videoBadge(photo)}
      <button class="album-remove mono" type="button" data-remove="${photo.image_id}" title="Remove from album">×</button>
    </div>
  `).join("");
  grid.querySelectorAll(".album-photo").forEach((tile) => {
    tile.addEventListener("click", (event) => {
      if (event.target.closest(".album-remove")) return;
      if (handleSelectClick(event, tile, "album",
          (i) => state.currentAlbum?.images?.[i]?.image_id)) return;
      openPhoto(Number(tile.dataset.imageId));
    });
  });
  grid.querySelectorAll(".album-remove").forEach((button) => {
    button.addEventListener("click", async () => {
      const album = state.currentAlbum;
      if (!album) return;
      try {
        await request(`/albums/${album.album_id}/items/remove`, {
          method: "POST",
          body: JSON.stringify({ image_ids: [Number(button.dataset.remove)] }),
        });
        openAlbum(album.album_id);
      } catch (error) {
        showError(`Could not remove the photo: ${error.message}`);
      }
    });
  });
}

async function loadAlbumSuggestions(albumId) {
  const row = $("albumSuggestions");
  row.innerHTML = '<div class="empty slim-empty">Looking for photos that fit…</div>';
  try {
    const body = await request(`/albums/${albumId}/suggestions?limit=18`);
    const suggestions = body.suggestions || [];
    if (!suggestions.length) {
      row.innerHTML = '<div class="empty slim-empty">Add a photo or two first — suggestions come from what the album already holds.</div>';
      return;
    }
    row.innerHTML = suggestions.map((photo) => `
      <div class="dupe-thumb suggest-thumb" data-image-id="${photo.image_id}">
        <img loading="lazy" src="${API}/images/${photo.image_id}/thumbnail?size=grid&format=webp" alt="">
        <button class="mini-btn yes add-suggest" type="button" data-add="${photo.image_id}" aria-label="Add to album">+</button>
      </div>
    `).join("");
    row.querySelectorAll(".suggest-thumb").forEach((thumb) => {
      thumb.addEventListener("click", (event) => {
        if (event.target.closest(".add-suggest")) return;
        openPhoto(Number(thumb.dataset.imageId));
      });
    });
    row.querySelectorAll(".add-suggest").forEach((button) => {
      button.addEventListener("click", async () => {
        const album = state.currentAlbum;
        if (!album) return;
        button.disabled = true;
        try {
          await request(`/albums/${album.album_id}/items`, {
            method: "POST",
            body: JSON.stringify({ image_ids: [Number(button.dataset.add)] }),
          });
          openAlbum(album.album_id);
        } catch (error) {
          showError(`Could not add the photo: ${error.message}`);
          button.disabled = false;
        }
      });
    });
  } catch {
    row.innerHTML = "";
  }
}

async function saveAlbumTitle() {
  const album = state.currentAlbum;
  if (!album) return;
  const name = $("albumTitle").value.trim();
  if (!name || name === album.name) return;
  try {
    await request(`/albums/${album.album_id}`, {
      method: "PATCH",
      body: JSON.stringify({ name }),
    });
    album.name = name;
    await loadAlbums();
  } catch (error) {
    showError(`Rename failed: ${error.message}`);
  }
}

async function deleteAlbum() {
  const album = state.currentAlbum;
  if (!album) return;
  if (!state.albumDeleteArmed) {
    state.albumDeleteArmed = true;
    $("albumDelete").textContent = "Really delete? Photos stay in the library";
    return;
  }
  try {
    await request(`/albums/${album.album_id}`, { method: "DELETE" });
    state.currentAlbum = null;
    loadAlbumsView();
  } catch (error) {
    showError(`Could not delete the album: ${error.message}`);
  }
}

function toggleAlbumPicker() {
  const panel = $("albumPickPanel");
  const show = panel.classList.contains("hidden");
  panel.classList.toggle("hidden", !show);
  $("modalAlbumBtn").setAttribute("aria-expanded", String(show));
  if (!show) return;
  loadAlbums().then(() => {
    $("albumPickList").innerHTML = state.albums.length
      ? state.albums.map((album) => `
          <button class="picker-row" type="button" data-album-id="${album.album_id}">
            <span class="picker-name">${escapeHtml(album.name)}</span>
            <span class="picker-count mono">${album.photo_count}</span>
          </button>`).join("")
      : '<div class="empty slim-empty">No albums yet — create one below.</div>';
    $("albumPickList").querySelectorAll(".picker-row").forEach((rowEl) => {
      rowEl.addEventListener("click", () =>
        addCurrentPhotoToAlbum(Number(rowEl.dataset.albumId)));
    });
  });
}

async function addCurrentPhotoToAlbum(albumId) {
  if (state.modalImageId == null) return;
  try {
    await request(`/albums/${albumId}/items`, {
      method: "POST",
      body: JSON.stringify({ image_ids: [state.modalImageId] }),
    });
    $("albumPickPanel").classList.add("hidden");
    $("modalAlbumBtn").textContent = "Added ✓";
    setTimeout(() => { $("modalAlbumBtn").textContent = "Add to album"; }, 1500);
  } catch (error) {
    showError(`Could not add to the album: ${error.message}`);
  }
}

// ---------------------------------------------------------------------------
// Paste an image to search
// ---------------------------------------------------------------------------

async function searchByImageFile(file) {
  clearError();
  setView("photos");
  $("photoGrid").innerHTML = skeletonGrid();
  startLoad();
  try {
    const form = new FormData();
    form.append("file", file, file.name || "pasted.png");
    form.append("per_page", String(state.perPage));
    const response = await fetch(`${API}/search/by-image`, {
      method: "POST",
      body: form,
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || `Search failed (${response.status})`);
    }
    state.similarTo = { pasted: true };
    setImageQuery({
      url: URL.createObjectURL(file),
      label: "pasted image",
      blob: true,
    });
    renderSelectedPeople();
    $("similarText").textContent = "Results for the image you pasted.";
    $("similarBanner").classList.remove("hidden");
    renderPhotos(body);
    $("resultCount").textContent =
      `${body.total.toLocaleString()} MATCHES · PASTED IMAGE`;
    $("pager").classList.add("hidden");
  } catch (error) {
    $("photoGrid").innerHTML = "";
    showError(`Image search failed: ${error.message}`);
  } finally {
    endLoad();
  }
}

function handlePaste(event) {
  const items = event.clipboardData?.items || [];
  for (const item of items) {
    if (item.kind === "file" && item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) {
        event.preventDefault();
        searchByImageFile(file);
      }
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// OCR status (Library tab)
// ---------------------------------------------------------------------------

async function loadOcrStatus() {
  try {
    const status = await request("/admin/ocr");
    const line = $("ocrStatusLine");
    if (!status.available) {
      $("ocrScan").disabled = true;
      $("ocrCopy").textContent =
        "The OCR engine is not installed in this build, so text inside photos cannot be read yet.";
      line.innerHTML = "";
      return;
    }
    $("ocrScan").disabled = Boolean(state.activeJob);
    const remaining = Math.max(0, status.total_images - status.scanned);
    $("ocrScan").textContent = remaining ? "Scan photo text" : "Rescan (up to date)";
    line.innerHTML = `<div class="model-line"><span>coverage</span>` +
      `<span>${status.scanned.toLocaleString()} of ${status.total_images.toLocaleString()} scanned` +
      ` · ${status.with_text.toLocaleString()} with text</span></div>`;
  } catch {
    $("ocrStatusLine").innerHTML = "";
  }
}

async function startOcrScan() {
  clearError();
  try {
    const job = await request("/admin/ocr", { method: "POST" });
    await monitorJob(job);
    loadOcrStatus();
  } catch (error) {
    showError(`Could not scan for text: ${error.message}`);
  }
}

// ---------------------------------------------------------------------------
// Curation backup (Library tab)
// ---------------------------------------------------------------------------

async function exportCuration() {
  clearError();
  const button = $("curationExport");
  button.disabled = true;
  try {
    const data = await request("/admin/curation");
    const blob = new Blob([JSON.stringify(data, null, 2)],
                          { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `photolib-backup-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 5000);
    $("curationStatus").innerHTML =
      `<div class="model-line"><span>exported</span>` +
      `<span>${data.people.length} people · ${data.albums.length} albums</span></div>`;
  } catch (error) {
    showError(`Backup failed: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

async function importCuration(file) {
  clearError();
  startLoad();
  try {
    const data = JSON.parse(await file.text());
    const report = await request("/admin/curation", {
      method: "POST",
      body: JSON.stringify(data),
    });
    const bits = [
      `${report.people_restored} people restored`,
      `${report.albums_created} albums created`,
      `${report.album_items_added} photos re-filed`,
    ];
    if (report.people_skipped) {
      bits.push(`${report.people_skipped} skipped — no clear match`);
    }
    $("curationStatus").innerHTML =
      `<div class="model-line"><span>restored</span><span>${bits.join(" · ")}</span></div>`;
    await Promise.all([loadPeople(), loadAlbums(), loadStats()]);
  } catch (error) {
    showError(`Restore failed: ${error.message}`);
  } finally {
    endLoad();
    $("curationFile").value = "";
  }
}

// ---------------------------------------------------------------------------
// Stats and startup
// ---------------------------------------------------------------------------

async function loadStats() {
  const stats = await request("/stats");
  const videos = stats.total_videos || 0;
  const parts = [
    `${((stats.total_images || 0) - videos).toLocaleString()} PHOTOS`,
  ];
  if (videos) parts.push(`${videos.toLocaleString()} VIDEOS`);
  parts.push(
    `${(stats.total_people || 0).toLocaleString()} PEOPLE`,
    `${(stats.total_faces || 0).toLocaleString()} FACES`,
  );
  $("topStats").textContent = parts.join(" · ");
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
  $("tabAlbums").addEventListener("click", () => setView("albums"));
  $("tabManage").addEventListener("click", () => setView("manage"));

  $("mergeToggle").addEventListener("click", () => {
    localStorage.setItem("photolib.mergeOpen", mergeReviewOpen() ? "0" : "1");
    applyMergeCollapse();
  });

  $("albumCreateForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = $("albumName").value.trim();
    if (!name) return;
    try {
      const album = await createAlbum(name);
      $("albumName").value = "";
      openAlbum(album.album_id);
    } catch (error) {
      showError(`Could not create the album: ${error.message}`);
    }
  });
  $("albumBack").addEventListener("click", loadAlbumsView);
  $("albumTitle").addEventListener("blur", saveAlbumTitle);
  $("albumTitle").addEventListener("keydown", (event) => {
    if (event.key === "Enter") saveAlbumTitle();
  });
  $("albumDelete").addEventListener("click", deleteAlbum);
  $("modalAlbumBtn").addEventListener("click", toggleAlbumPicker);
  $("albumPickCreate").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = $("albumPickName").value.trim();
    if (!name || state.modalImageId == null) return;
    try {
      await createAlbum(name, [state.modalImageId]);
      $("albumPickName").value = "";
      $("albumPickPanel").classList.add("hidden");
      $("modalAlbumBtn").textContent = "Added ✓";
      setTimeout(() => { $("modalAlbumBtn").textContent = "Add to album"; }, 1500);
    } catch (error) {
      showError(`Could not create the album: ${error.message}`);
    }
  });

  $("ocrScan").addEventListener("click", startOcrScan);
  document.addEventListener("paste", handlePaste);

  $("searchForm").addEventListener("submit", (event) => {
    event.preventDefault();
    search(1);
  });
  $("sortSelect").addEventListener("change", () => search(1));
  $("dateFrom").addEventListener("change", () => search(1));
  $("dateTo").addEventListener("change", () => search(1));
  $("cameraFilter").addEventListener("change", () => search(1));
  $("mediaFilter").addEventListener("change", () => search(1));
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
  $("modalPrev").addEventListener("click", () => navigatePhoto(-1));
  $("modalNext").addEventListener("click", () => navigatePhoto(1));
  $("untaggedReview").addEventListener("click", () => {
    state.untaggedOnly = true;
    setView("photos");
    renderSelectedPeople();
    search(1);
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

  $("addRootButton").addEventListener("click", addRootFlow);
  $("rescanAllButton").addEventListener("click", rescanAll);
  $("findDupes").addEventListener("click", loadDupes);

  $("curationExport").addEventListener("click", exportCuration);
  $("curationImportBtn").addEventListener("click", () => $("curationFile").click());
  $("curationFile").addEventListener("change", () => {
    const file = $("curationFile").files?.[0];
    if (file) importCuration(file);
  });

  $("selClear").addEventListener("click", clearSelection);
  $("selTrash").addEventListener("click", batchTrash);
  $("selExport").addEventListener("click", batchExport);
  $("selRemoveAlbum").addEventListener("click", batchRemoveFromAlbum);
  $("selAlbumBtn").addEventListener("click", () => toggleSelectionAlbumPanel());
  $("selAlbumCreate").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = $("selAlbumName").value.trim();
    if (!name) return;
    try {
      const album = await createAlbum(name);
      $("selAlbumName").value = "";
      await batchAddToAlbum(album.album_id);
    } catch (error) {
      showError(`Could not create the album: ${error.message}`);
    }
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest("#selectionBar")) toggleSelectionAlbumPanel(false);
  });
  initZoom();

  $("shortcutClose").addEventListener("click", () => toggleShortcuts(false));
  $("shortcutModal").addEventListener("click", (event) => {
    if (event.target === $("shortcutModal")) toggleShortcuts(false);
  });
  $("modalVideo").addEventListener("error", () => {
    // Fires for real decode failures (e.g. HEVC without the Windows codec)
    // and also when a source is detached — dataset.imageId separates them.
    const imageId = $("modalVideo").dataset.imageId;
    if (!imageId) return;
    $("videoFallbackLink").href = `${API}/images/${imageId}?download=true`;
    $("videoFallback").classList.remove("hidden");
  });

  document.addEventListener("keydown", (event) => {
    const photoOpen = !$("photoModal").classList.contains("hidden");
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
    if (event.key === "Escape") {
      // Close only the topmost layer, so backing out of a photo opened
      // from a person still leaves the person open.
      if (!$("shortcutModal").classList.contains("hidden")) toggleShortcuts(false);
      else if (photoOpen) closePhoto();
      else if (!$("personModal").classList.contains("hidden")) closePerson();
      else if (!$("selAlbumPanel").classList.contains("hidden")) {
        toggleSelectionAlbumPanel(false);
      } else if (state.selection.size) clearSelection();
      else togglePeoplePanel(false);
      return;
    }
    if (typing) return; // every other key below is a shortcut, not input

    if (event.key === "?") {
      event.preventDefault();
      toggleShortcuts();
      return;
    }
    if (event.key === "/" && !photoOpen) {
      event.preventDefault();
      setView("photos");
      $("searchInput").focus();
      return;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
      if (photoOpen) return;
      const albumOpen = state.view === "albums"
        && !$("albumDetail").classList.contains("hidden");
      if (state.view === "photos" || albumOpen) {
        event.preventDefault();
        selectAllVisible();
      }
      return;
    }
    if (event.key === "Delete") {
      if (photoOpen) trashOpenPhoto();
      else if (state.selection.size) batchTrash();
      return;
    }
    if (!photoOpen) return;
    if (event.key === "ArrowRight") navigatePhoto(1);
    if (event.key === "ArrowLeft") navigatePhoto(-1);
    if (event.target.tagName === "VIDEO") return; // native controls own these
    if (event.key === " " && modalVideoActive()) {
      event.preventDefault();
      const video = $("modalVideo");
      if (video.paused) video.play();
      else video.pause();
      return;
    }
    if (!modalVideoActive()) {
      const rect = $("modalStage").getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      if (event.key === "+" || event.key === "=") zoomTowards(cx, cy, 1.3);
      if (event.key === "-" || event.key === "_") zoomTowards(cx, cy, 1 / 1.3);
      if (event.key === "0") resetZoom();
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
