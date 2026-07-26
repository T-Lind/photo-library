# Actionable library round — design

Date: 2026-07-26 · Branch: feature/actionable-library

## Why

The library is read-only: you can find anything but act on nothing. This
round adds the acting: select photos, put them in the Recycle Bin, zoom
into them, copy the text out of them, and back up the curation work
(names, albums) that currently lives only inside the LanceDB data dir.
Plus one search QoL: show the query image when searching by picture.

## Multi-select + batch actions

Selection lives in the UI as a `Set` of image_ids that survives paging.
Ctrl/Cmd-click toggles, Shift-click extends a range within the current
page, and a hover check-circle on each tile toggles without opening the
photo. A floating bar appears while anything is selected: count, "Add to
album" (album chooser + create-new), "Remove from album" (only when the
selection was made inside an album), "Trash", "Clear". Escape clears the
selection before it closes anything else. Plain click still opens the
photo — selection never gets in the way of browsing.

## Trash — Recycle Bin, never rm

`POST /images/trash {image_ids}` (≤500 per call) sends the original files
to the OS Recycle Bin via `send2trash`, then removes every trace from the
library: image/face/OCR rows, album memberships, cached thumbnails. People
whose faces were in the trashed photos are recomputed (centroid, count,
cover). A file already missing from disk is still cleaned out of the
index. Nothing is ever permanently deleted by this app — recovery is the
OS Recycle Bin.

The duplicates panel becomes actionable: each group shows per-photo file
sizes and a two-step "Keep largest, trash the rest" button.

## Zoom & pan in the viewer

Wheel zooms toward the cursor (1×–8×), drag pans while zoomed,
double-click toggles fit ↔ 2.5× at the cursor. Zoom resets on photo
change. Pure CSS transform on the existing `<img>` — no new layout.

## Copy text from a photo

`GET /images/{id}/text` returns the full stored OCR text (details keeps
its 800-char excerpt). The details panel gets a "Copy text" button that
fetches the full text into the clipboard — the whole screenshot history
becomes copy-able.

## Curation backup / restore

The irreplaceable data is what the user did by hand: person names, hidden
flags, album membership. `GET /admin/curation` exports it as JSON keyed by
`content_hash` (already stored per image), so the backup survives moves,
renames, and full re-indexes. `POST /admin/curation` restores:

- **Albums: exact.** Created by name if missing; members matched by
  content_hash are re-added.
- **People names: best-effort.** A rebuilt library has different
  person_ids and cluster boundaries, so each exported person is matched to
  the current person with the greatest photo overlap (by content_hash);
  the name and hidden flag are applied when at least half of the exported
  photos land on one current person. Ambiguous or weak matches are
  skipped and reported, never guessed.

Out of scope: face-level confirmed assignments (not reconstructible once
face_ids change). The Library tab gets a Backup panel: export downloads
the JSON, restore takes a file.

## Query-image chip

Searching by picture (Ctrl+V paste or "More like this") now pins a chip
with a thumbnail of the query image next to the other filter chips, so
you can see what you searched by; × clears it back to normal browsing.

## Testing

Service-level: trash removes all traces + recomputes people (send2trash
monkeypatched), missing file still cleaned, album items purged; curation
export→wipe→import round-trips albums exactly and re-names people by
overlap; full-text endpoint; duplicates carry file sizes. UI stays
untested as before (no JS harness).
