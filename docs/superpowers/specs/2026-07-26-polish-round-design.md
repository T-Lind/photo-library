# Polish round: multi-person search, folders, EXIF, redesign

Date: 2026-07-26 · Branch: codex/windows-installer (PR #6)

## Features

- **Multi-person search** — the person filter is now a multi-select picker
  with an all/any mode (backend `people_ids` + `people_mode` already
  existed). Selected people render as removable chips.
- **Faces in the photo viewer** — every detected face shows as a labelled
  crop under the photo; clicking a known face offers *Open profile* and
  *Their photos*.
- **Loading states** — a cyanotype "exposure strip" sweeps under the top bar
  during any search, and grids show shimmering contact-sheet skeleton frames.
- **Library folders** (new backend) — source folders persist in
  `state_dir/roots.json` (`GET/POST/DELETE /admin/roots`); indexing a folder
  registers it automatically. The Library tab lists them with counts,
  per-folder rescan, rescan-all (sequential jobs), and forget.
- **EXIF** — a Details panel in the viewer (taken/camera/size/file/folder/
  GPS), plus a camera filter (`Filters.camera`, `GET /cameras`; the select
  hides itself when no photo has camera EXIF) and a has-location toggle.
- **Timeline** — clickable photos-per-month histogram; a click sets the date
  range to that month.
- **Duplicates** — Library tab surfaces the existing `/duplicates` endpoint
  as grouped thumbnails.

## Redesign

Graphite contact-sheet look: near-monochrome chrome so photographs carry the
color; cyanotype-blue accent (`#6fa0ff`/`#2c4e9e`) for interactive states
only. Native Windows type — Segoe UI Variable for text, Cascadia Mono for
frame-marking metadata (counts, dates, labels in tracked uppercase). No
external fonts or requests: the app's premise is that nothing leaves the
machine. Signature elements: the exposure-strip loader, hover frame numbers
on grid tiles, and mono stat lines. Reduced-motion respected; focus rings
visible.

## Testing

Roots add/list/remove + validation and camera filter/listing tests in
`tests/test_api.py`; full suite passes; UI verified by driving the real
library in Chrome (picker, faces strip, EXIF, Library tab, timeline).
