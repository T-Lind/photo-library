// Curation controls share the same local API and application state.
let savedSearchItems = [];

function initCuration() {
  $("favoritesOnly").addEventListener("change", filterChanged);
  $("minRating").addEventListener("change", filterChanged);
  $("saveSearchForm").addEventListener("submit", async event => {
    event.preventDefault();
    try {
      await request("/saved-searches", {method:"POST", body:JSON.stringify({
        name:$("savedSearchName").value, request:{...currentFilters(), ...state.savedFilterExtras,
          query:$("searchInput").value || null, sort:$("sortSelect").value}})});
      $("savedSearchName").value = "";
      await loadSavedSearches();
    } catch (error) { showError(error.message); }
  });
  $("savedSearchSelect").addEventListener("change", applySavedSearch);
  $("deleteSavedSearch").addEventListener("click", async () => {
    const id = $("savedSearchSelect").value;
    if (!id) return;
    try { await request(`/saved-searches/${id}`, {method:"DELETE"}); await loadSavedSearches(); }
    catch (error) { showError(error.message); }
  });
  $("modalFavorite").addEventListener("click", () => updateAnnotation({favorite:$("modalFavorite").getAttribute("aria-pressed") !== "true"}));
  $("modalRating").addEventListener("change", () => updateAnnotation({rating:Number($("modalRating").value)}));
  $("qualityScan").addEventListener("click", () => curationJob("/admin/quality"));
  $("retryFiles").addEventListener("click", () => curationJob("/admin/retry"));
  $("modalBurst").addEventListener("click", compareBurst);
  $("burstClose").addEventListener("click", () => $("burstDialog").close());
}

async function loadSavedSearches() {
  try {
    savedSearchItems = (await request("/saved-searches")).searches;
    $("savedSearchSelect").innerHTML = '<option value="">Saved searches…</option>' + savedSearchItems.map(s =>
      `<option value="${escapeHtml(s.id)}">${escapeHtml(s.name)}</option>`).join("");
  } catch (error) { showError(`Saved searches: ${error.message}`); }
}

function applySavedSearch() {
  const saved = savedSearchItems.find(s => s.id === $("savedSearchSelect").value);
  if (!saved) return;
  const r = saved.request;
  $("searchInput").value = r.query || "";
  $("dateFrom").value = (r.start_date || "").slice(0,10);
  $("dateTo").value = (r.end_date || "").slice(0,10);
  $("cameraFilter").value = r.camera || "";
  $("mediaFilter").value = r.media || "";
  $("locationToggle").checked = r.has_location === true;
  $("favoritesOnly").checked = r.favorites_only;
  $("minRating").value = String(r.min_rating);
  $("sortSelect").value = r.sort;
  $("searchMode").value = r.search_mode || "both";
  syncSearchModeUi();
  // Copy, don't alias: the picker mutates this array, which would silently
  // edit the saved search in memory.
  state.selectedPeople = [...(r.people_ids || [])];
  state.peopleMode = r.people_mode || "any";
  // Keep the visible all/any radio in step with the restored mode.
  const modeRadio = document.querySelector(
    `input[name="peopleMode"][value="${state.peopleMode}"]`);
  if (modeRadio) modeRadio.checked = true;
  state.untaggedOnly = r.untagged_only;
  state.near = r.near_lat == null ? null : {lat:r.near_lat, lon:r.near_lon, km:r.near_km};
  state.savedFilterExtras = {folder:r.folder, has_faces:r.has_faces};
  renderSelectedPeople();
  search(1);
}

function renderPhotoCuration(details) {
  $("modalFavorite").setAttribute("aria-pressed", String(Boolean(details.favorite)));
  $("modalFavorite").textContent = details.favorite ? "♥ Favorite" : "♡ Favorite";
  $("modalRating").value = String(details.rating || 0);
  const ranked = state.results.find(r => r.image_id === details.image_id);
  $("modalQuality").textContent = (ranked?.quality_reasons?.length ? ranked.quality_reasons : details.quality_reasons || []).join(" · ");
}

async function updateAnnotation(value, imageId=state.modalImageId) {
  if (imageId == null) return;
  try {
    const updated = await request(`/images/${imageId}/annotation`, {method:"PATCH", body:JSON.stringify(value)});
    const photo = state.results.find(r => r.image_id === imageId);
    if (photo) Object.assign(photo, updated);
    if (state.modalImageId === imageId) {
      $("modalFavorite").setAttribute("aria-pressed", String(updated.favorite));
      $("modalFavorite").textContent = updated.favorite ? "♥ Favorite" : "♡ Favorite";
      $("modalRating").value = String(updated.rating);
    }
    const tile = document.querySelector(`#photoGrid [data-image-id="${imageId}"]`);
    if (tile) {
      tile.querySelector(".curation-badge")?.remove();
      if (updated.favorite || updated.rating) {
        const badge = document.createElement("span"); badge.className = "curation-badge";
        badge.textContent = `${updated.favorite ? "♥ " : ""}${"★".repeat(updated.rating)}`;
        tile.appendChild(badge);
      }
    }
    return updated;
  } catch (error) { showError(error.message); }
}

async function curationJob(path) {
  try { await monitorJob(await request(path, {method:"POST"})); await loadFailures(); }
  catch (error) { showError(error.message); }
}

async function loadFailures() {
  try {
    const files = (await request("/admin/failures")).files;
    $("failedFiles").innerHTML = files.length ? files.slice(0,100).map(f =>
      `<div class="failure-row"><strong>${escapeHtml(f.path)}</strong><span>${escapeHtml(f.error)}</span></div>`).join("") +
      (files.length > 100 ? `<p>${files.length} files pending; showing the first 100. Retry processes all.</p>` : "") : "No failed files.";
    $("retryFiles").disabled = !files.length;
  } catch (error) { showError(error.message); }
}

async function relocateFolder(root) {
  const destination = await chooseFolder(false);
  if (!destination) return;
  const panel = $("relocationReview");
  panel.classList.remove("hidden");
  panel.textContent = "Verifying every photo at the new location…";
  try {
    const plan = await request("/admin/roots/relocate", {method:"POST", body:JSON.stringify({old_folder:root.path,new_folder:destination})});
    panel.innerHTML = `<p>${plan.photos} photos verified at ${escapeHtml(destination)}. Photo IDs, albums and corrections will be preserved.</p>
      <button class="btn primary" type="button">Use this location</button> <button class="btn ghost" type="button">Cancel</button>`;
    const [apply, cancel] = panel.querySelectorAll("button");
    cancel.onclick = () => panel.classList.add("hidden");
    apply.onclick = async () => {
      apply.disabled = true;
      try {
        await request("/admin/roots/relocate", {method:"POST", body:JSON.stringify({old_folder:root.path,new_folder:destination,verification:plan.verification})});
        panel.textContent = "Location updated. Your library is ready.";
        await loadRoots(); await loadSavedSearches();
      } catch (error) { showError(error.message); apply.disabled = false; }
    };
  } catch (error) { panel.textContent = error.message; }
}

async function previewRestore(data) {
  const check = await request("/admin/curation/verify", {method:"POST", body:JSON.stringify(data)});
  const panel = $("curationStatus");
  panel.innerHTML = `<p>Backup verified. ${check.faces_matched} face corrections match; ${check.faces_unmatched} missing or ambiguous; ${check.identity_conflicts} identity conflicts. ${check.annotations_matched} photo ratings/favorites match.</p>
    <button class="btn primary" type="button">Restore matched curation</button> <button class="btn ghost" type="button">Cancel</button>`;
  const [apply, cancel] = panel.querySelectorAll("button");
  cancel.onclick = () => { panel.textContent = "Restore cancelled."; };
  apply.onclick = async () => {
    apply.disabled = true;
    try {
      const report = await request("/admin/curation", {method:"POST", body:JSON.stringify(data)});
      panel.textContent = `Restored ${report.faces_matched} exact face corrections, ${report.annotations_matched} ratings/favorites and ${report.albums_created} albums. ${report.faces_unmatched} corrections unmatched; ${report.identity_conflicts} identity conflicts retained for review.`;
      await Promise.all([loadPeople(),loadAlbums(),loadSavedSearches(),loadStats()]);
    } catch (error) { showError(error.message); apply.disabled = false; }
  };
}

async function compareBurst() {
  const imageId = state.modalImageId;
  try {
    const result = await request(`/images/${imageId}/burst`);
    $("burstSummary").textContent = `${result.total || result.images.length} nearby, visually similar shots. Suggestions keep every original. Compare at the same scale below.`;
    $("burstGrid").innerHTML = result.images.map(photo => `<article class="burst-candidate">
      <img src="${API}/images/${photo.image_id}/thumbnail?size=large" alt="${escapeHtml(photo.filename)}">
      <strong>${escapeHtml(photo.filename)}</strong>
      <p>${photo.image_id === result.suggested_keeper ? "Suggested keeper · " : ""}${escapeHtml((photo.quality_reasons || []).join(" · "))}</p>
      <button class="btn" type="button" data-keeper="${photo.image_id}">${photo.favorite ? "♥ Favorite" : "Mark as favorite"}</button>
      <button class="btn ghost" type="button" data-open="${photo.image_id}">Inspect full size</button></article>`).join("");
    $("burstGrid").querySelectorAll("[data-keeper]").forEach(button => button.onclick = async () => {
      if (await updateAnnotation({favorite:true}, Number(button.dataset.keeper))) button.textContent = "♥ Favorite";
    });
    $("burstGrid").querySelectorAll("[data-open]").forEach(button => button.onclick = () => {
      $("burstDialog").close(); openPhoto(Number(button.dataset.open));
    });
    $("burstDialog").showModal();
  } catch (error) { showError(error.message); }
}
