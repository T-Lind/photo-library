# Local OCR, albums, and paste-to-search — design

Date: 2026-07-26 · Branch: feature/local-ocr (PR #7)

## Why OCR

The library is screenshot-heavy, and queries like "JROTC" are literal
strings rendered inside images. The 224px embedding model cannot read fine
print, so semantic ranking underperforms exactly where OpenAI CLIP
(accidentally good at typography) used to shine. OCR closes the gap with
exact text matching.

## Engine

Two backends behind one interface (`photolib/ocr.py`), chosen by
`PHOTO_OCR_BACKEND=auto`:

1. **Windows.Media.Ocr** (via winsdk) — ships with Windows, no model
   download, benchmarked at ~0.25s per dense 2,544px screenshot.
2. **RapidOCR** (PaddleOCR on onnxruntime) — cross-platform fallback,
   models inside the wheel, ~5–8s per dense screenshot on CPU.

Both are strictly offline. Neither is required: without them the app
behaves exactly as before. A `stub` backend (filename words) keeps CI
model-free.

## Storage: no reindex, ever

Text lives in a new lazily-created `ocr` table (image_id, text, engine,
updated_at) — adding OCR to an existing library is an append, not a schema
migration. A row exists for every *scanned* image even when empty, which is
what lets the backfill know what remains.

- **Index time**: the indexer reuses the already-decoded pixel buffer
  (decode-once holds) and writes OCR rows alongside image rows.
- **Backfill** (`POST /admin/ocr`): scans only never-scanned images, newest
  first, in a cancellable, resumable background job. The images table is
  never rewritten (regression-tested via table version).

## Search

Hybrid ranking in `PhotoService.search`: photos whose text contains every
query token rank ahead of semantic matches (marked `text_match`, shown as a
TEXT badge). Date sorts merge both sets. The browse index holds lowercase
text in memory (~KBs per screenshot; empty for ordinary photos) and tracks
the OCR table version for freshness.

## Albums

`albums` + `album_items` tables (lazy). CRUD under `/albums`; the album's
"looks like it belongs" strip ranks non-members against the mean embedding
of the newest 32 members — one-click add. UI: Albums tab (create, inline
rename, two-step delete, hover-remove) and an "Add to album" picker (with
create-new) in the photo viewer.

## QoL

- Ctrl+V an image anywhere → `/search/by-image` on the pasted bytes.
- Merge review collapsed by default with a count badge; state remembered.
- `/` focuses search.

## Out of scope

Tantivy/FTS indexing (in-memory substring is fine below ~200k screenshots),
OCR language packs beyond the user profile's, translation.
