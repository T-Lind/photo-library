# Videos, shortcuts, and getting things out — Round 7 design

Approved bundle: (1) video support end to end, (2) keyboard shortcuts with a
discoverable cheat sheet, (3) copy image / Show in Explorer / export copies,
(4) AVIF. Everything stays local and offline.

## 1. Video support

### The one structural decision

Videos are **rows in the existing `images` table**, not a parallel system.
A video is represented by its *poster frame* — a frame pulled from ~1s in —
and that frame flows through the exact same pipeline as a photo: SigLIP
embedding (semantic search), face detection (people), perceptual hash
(duplicates), OCR, thumbnails. Two new columns describe what makes it a
video: `media_type` ("image" | "video") and `duration_ms`.

This buys, for free: search, people, albums, trash, duplicates, curation
backup, multi-select — every existing feature works on videos with no new
code paths.

### Decode layer (`photolib/imageio.py`)

- `VIDEO_EXTS = {.mp4 .mov .m4v .webm .mkv .avi .3gp .mpg .mpeg .wmv .mts .m2ts}`,
  included in `SUPPORTED_EXTS` only when `imageio-ffmpeg` imports; otherwise
  videos are skipped with one warning (same pattern as pillow-heif).
- `open_image(path)` grows a video branch: seek to `min(1s, duration/2)`,
  decode one frame via `imageio_ffmpeg.read_frames`, return it as a PIL
  image capped at 1920px long edge. A small keyed-by-`(path, mtime)` LRU
  (8 entries) absorbs the fact that `_prepare` + thumbnails decode the same
  file several times; cache returns copies so callers may close them.
- `probe_video(path)` → `{duration_ms, width, height}` from the ffmpeg meta
  dict (no frame decode); LRU-cached the same way. `read_size` uses it.
- ffmpeg auto-rotates .mov display-matrix rotation during decode, so poster
  orientation is correct without EXIF handling.

### Metadata (`photolib/exif.py`)

`read_metadata` gets a video branch: parse `creation_time` from
`ffmpeg -i` stderr (validated: `creation_time   : 2023-05-17T14:30:00.000000Z`).
The existing fallback chain (filename date, year sanity check) applies
unchanged. GPS in videos is out of scope this round.

### Schema and migration (`photolib/db.py`)

- `images_schema` gains `media_type` (string) and `duration_ms` (int64).
- `Library.ensure_media_columns()`: if `media_type` missing from the images
  table schema, `add_columns({"media_type": "'image'", "duration_ms": ...})`
  — an in-place backfill, no re-embed, no version bump (same philosophy as
  the OCR table). Called from `PhotoService.__init__` and
  `Indexer.index_directory`; `LibraryIndex._rebuild` additionally tolerates
  the columns being absent so a read-only open of an old library never dies.
- The user's real library needs **no re-index**: next "Index new files" run
  simply discovers videos as new files.

### Browse and API

- `Filters.media: Optional[str]` ("image"/"video"); `LibraryIndex` keeps a
  `media_type` array; the filter is one boolean mask. Search request schema
  passes it through.
- `IMAGE_LIST_COLUMNS` += `media_type`, `duration_ms` → every result row and
  `image_details` carry them automatically.
- `GET /images/{id}` already serves the original through Starlette's
  `FileResponse`, which supports HTTP Range (installed starlette 1.3.1), so
  `<video>` seeking works. Add video MIME types to `MIME_BY_SUFFIX`.
- Stats gain a `videos` count.

### Player UI

- Tiles: ▶ badge + `m:ss` duration overlay for `media_type === "video"`.
- Viewer: videos render a native `<video controls autoplay>` element (same
  stage as the zoom container; zoom applies to images only). Native controls
  give play/pause/scrub/volume/fullscreen. `canplay`/`error` events decide
  the fallback: on decode failure (e.g. HEVC without the Windows codec),
  show the poster thumbnail with an explanatory note and a "Download /
  open in your player" link to the original-file URL.
- Filter bar: "Videos" toggle chip → `media` filter.

### Packaging

`imageio-ffmpeg` added to requirements (min + desktop); PyInstaller spec
collects the wheel's bundled `ffmpeg-win-x86_64*.exe` via
`collect_data_files("imageio_ffmpeg")`. ~31MB — the price of playing half
of everyone's camera roll.

## 2. Keyboard shortcuts + cheat sheet

| Key | Context | Action |
| --- | --- | --- |
| `Delete` | selection active, or viewer open | two-step trash (arm → confirm), same flow as the button |
| `Ctrl+A` | photos grid or album open | select everything on the current page |
| `+` / `-` / `0` | viewer, image | zoom in / out / reset |
| `Space` | viewer, video | play/pause |
| `?` | anywhere | shortcut cheat-sheet overlay |
| existing | | `/` search, `←`/`→` navigate, `Esc` close/clear |

The `?` overlay is a small modal listing all of the above; Escape closes it
first (it joins the front of the existing Escape priority chain). Shortcuts
never fire while an input/textarea has focus.

## 3. Getting things out

- **Copy image** (viewer): fetch the preview-size JPEG, draw to canvas,
  `ClipboardItem` PNG. Works for HEIC/RAW too because the preview endpoint
  re-encodes. Button next to Copy text with the same "Copied ✓" feedback.
- **Show in Explorer** (viewer): `POST /images/{id}/reveal` runs
  `explorer /select,<path>`. Desktop-only affordance, local-only endpoint;
  module-level `_reveal_in_explorer` seam so tests stub it (the
  `_send_to_trash` pattern).
- **Export copies** (selection bar): `POST /images/export` with
  `{image_ids, dest}`; dest comes from the existing native folder picker
  endpoint. `shutil.copy2` (keeps timestamps); name collisions get
  ` (2)`-style suffixes; response reports `{copied, missing, failed}`.
  Originals are never touched or moved.

## 4. AVIF

Pillow 12 decodes AVIF natively (`features.check("avif")` is true in this
venv). Add `.avif` to `RASTER_EXTS` gated on that check, plus the MIME
entry. Nothing else changes.

## Testing

- **Video fixture**: `imageio_ffmpeg.write_frames` synthesizes tiny mp4s in
  CI (validated) — solid-color frames keyed to filename words so the stub
  embedder convention still holds; a `creation_time` tag tests date extraction.
- Tests: video indexed with correct media_type/duration/taken_at; searchable
  via stub; thumbnail generated from poster; media filter; Range request on
  the file endpoint returns 206; trash/albums work on a video row.
- Migration test: build a library with the old schema (drop the two columns)
  → `ensure_media_columns` backfills, browse works, old rows are "image".
- reveal/export: stub seam records calls; export collision-rename covered;
  unknown ids and missing files counted, not fatal.
- AVIF: generated `.avif` indexes end to end (skipped if Pillow lacks codec).
- UI: `node --check`; live smoke test on the real library.
