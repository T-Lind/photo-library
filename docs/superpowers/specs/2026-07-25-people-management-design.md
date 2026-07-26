# People management, filters, and desktop UI overhaul — design

Date: 2026-07-25
Branch: codex/windows-installer

## Goal

The packaged desktop app works, but its UI exposes a fraction of what the
backend can do. The user asked for: naming and merging people (with a UI that
shows both people's faces), suggested merges, more filters (date etc.),
optional text search, search history, and the guarantee that a merged person
keeps matching newly indexed photos.

## What already exists (no work needed)

- **Optional text search**: `POST /api/v1/search` treats `query: null` as a
  plain browse with filters.
- **Filters**: date range, people (any/all), has_location, has_faces, folder,
  untagged — all in `Filters` / `SearchRequest`, answered from the in-memory
  browse index.
- **People operations**: rename, hide, merge, delete, assign/detach faces
  (confirmed flag protects manual corrections from reclustering), per-person
  "is this also X?" face suggestions.
- **Merged people persist**: `merge_people` recomputes the target centroid
  from all of its faces, and `FaceAssigner` reloads people from the table at
  the start of every index run, so new photos match the merged identity.
  (Covered by a new regression test.)

## Backend additions

### 1. `GET /people/{person_id}/faces`

`PhotoService.person_faces(person_id, limit=200)` — the person's face rows
(face_id, image_id, bbox, quality, confirmed), best quality first. Needed by
the person detail view (see their faces, select wrong ones, detach) and the
merge review UI (see both people's faces side by side).

### 2. `GET /people/merge-suggestions`

`PhotoService.merge_suggestions(limit=20, min_similarity=None)`:

- Load visible (non-hidden) people with a valid centroid.
- Pairwise cosine similarity of centroids (they are stored L2-normalised, so
  this is one matrix product; a few thousand people is trivial).
- **Co-occurrence veto**: if two people appear together in at least one
  photo, they are almost certainly different people — never suggest that
  pair. Uses the browse index's per-person row sets.
- Keep pairs with similarity ≥ `min_similarity` (default:
  `settings.face_cluster_threshold`), sorted by similarity, top `limit`.
- Each pair is returned with a suggested direction: the *target* (identity
  to keep) is the named one; if both or neither are named, the one with more
  photos. Response rows: `{source: PersonOut-ish, target: PersonOut-ish,
  similarity}`.

Route is declared **before** `/{person_id}` so FastAPI does not try to parse
`merge-suggestions` as an int.

### 3. Suggestion rejection that sticks

`suggest_for_person` currently proposes any unassigned face near the
centroid. Rejecting one in the UI would bring it back next visit. Change the
filter to `person_id = UNASSIGNED AND confirmed = false`: since
`detach_faces` marks faces `confirmed = true`, the existing detach endpoint
becomes the persistence mechanism for "not this person / stop suggesting".
Such faces remain visible in the unassigned queue, so nothing is lost.

## Desktop UI overhaul (`desktop/ui`, dependency-free vanilla JS)

Two views behind tabs, shown once the library is ready: **Photos** and
**People**. State in one `state` object; view switching toggles sections.

### Photos view

- Search bar unchanged (text optional — placeholder says so), plus a filter
  row: date from / date to (`<input type="date">`), person select, sort
  select, and a Clear button. Any filter change re-runs the search.
- **Recent searches**: last 8 non-empty queries in
  `localStorage["photolib.recentSearches"]`, rendered as chips under the
  search bar; click re-runs, Clear wipes.
- Result meta line: "N photos · X ms".
- **Photo modal**: adds people chips (named via `/images/{id}/details`),
  camera/place/date line, "More like this" (renders `/images/{id}/similar`
  results into the grid with a dismissible banner), and a person chip click
  filters the library to that person.

### People view

- **Suggested merges strip** (from `/people/merge-suggestions`): each card
  shows both cover crops, names, photo counts, similarity, and
  `Merge` / `Not the same` buttons. `Merge` calls `POST /people/merge`
  (suggested direction). `Not the same` dismisses the pair locally
  (`localStorage["photolib.dismissedMerges"]`, key `min:max`).
- **People grid**: card per person — cover face crop
  (`/faces/{cover_face_id}/crop`), name or "Person N", photo count. Click
  opens the person panel.
- **Person panel** (modal):
  - Name input + Save (rename), so naming is one click away.
  - Actions: "Merge into…" (inline picker listing other people with a text
    filter; confirm merges this person into the chosen one), Hide/Unhide,
    "Forget person" as a two-step button (no native confirm dialogs).
  - Faces grid (`/people/{id}/faces`): multi-select crops, "Not this person"
    detaches selection.
  - Suggestions strip (`/people/{id}/suggestions`): per-face ✓ assign to
    this person / ✗ reject (detach → never auto-suggested again).
  - "Show photos" switches to Photos view filtered to this person.

### Error handling

All actions go through the existing `request()` helper and surface failures
in the existing error panel. Merge/detach/rename refresh people + photos so
counts and chips stay consistent.

## Testing

- API tests (stub backends, real LanceDB, existing fixtures):
  - person faces endpoint returns that person's faces only.
  - merge-suggestions proposes a manufactured split identity (detach one of
    a person's faces, assign it to a new person → centroids match).
  - co-occurrence veto: a split whose photo also contains the original
    person is not suggested.
  - rejected suggestion (detached, confirmed) disappears from suggestions.
- Indexer regression: split → merge → index a new photo of the person →
  the new face lands on the merged person_id.
- UI is dependency-free; syntax-check `app.js` with `node --check`.

## Out of scope

Multi-person filter chips ("all of"), map view, editing EXIF, backend-stored
search history (localStorage is enough for a single-user desktop app).
