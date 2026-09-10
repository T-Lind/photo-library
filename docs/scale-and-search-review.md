# Scale, multiple drives, and finding good photos

Review date: 2026-09-05. This is a code review with synthetic validation, not
a certification against a real 200k-photo collection.

The foundation is useful for a large personal library: vector search, columnar
browsing, page-sized metadata reads, incremental ingest, sharded thumbnails,
people correction, OCR, albums, exports, and multiple source folders already
exist. Million-photo and intermittently connected drive workflows need more work.

**Changes made during this review**

- Abort discovery on directory traversal errors before destructive writes.
  Reject `limit` combined with pruning. Confirm that an omitted file is absent
  before pruning, so missing codec support or directory exclusions do not remove
  existing files from the catalog. Other roots are retained.
- Serve cached thumbnails and face crops when originals are missing. Original
  delivery and uncached previews still report that the source is unavailable.
  Cache freshness resumes normally when a drive reconnects.
- Rank filtered subsets of at most 4096 photos exactly by cosine similarity,
  fetching vectors by indexed image ID. Empty person selections that match no
  photos avoid loading the text model. Broader ANN searches are unchanged.
- Correct the README's memory claim: it excluded Python containers and strings.

**Implemented after the review**

- Safe reprocessing prepares and validates replacements before a recoverable
  multi-table commit. It retains photo IDs and `added_at`, album membership,
  annotations, and matched confirmed face corrections. An ambiguous correction
  keeps the previous catalog row and enters the persistent retry queue.
- Numeric IDs are never reused after deletion. A write-ahead table-version
  journal rolls back interrupted multi-table ingest, replacement, relocation,
  and restore commits when the catalog reopens.
- Sources now have stable IDs. Relocation hashes every destination file, shows
  a review step, detects intervening catalog changes, then updates paths while
  retaining photo identity. Saved folder searches and failed-file paths follow.
- Favorites, ratings, saved searches, persistent failure retry, burst comparison,
  versioned quality measurements, and “Best quality + match” are in the API and
  desktop UI. Semantic relevance is weighted 75%; technical quality is 25%.
  Without a query, technical quality determines the order.
- Version 2 curation backups are checksummed and include exact confirmed face
  corrections, identity splits, confirmed unassigned faces, albums, annotations,
  saved searches, and roots. Restore provides a verification preview and reports
  unmatched or conflicting items. Original files, the complete generated index,
  caches, and model weights remain outside this curation backup.

**Highest-priority remaining scale work**

| Priority | Finding in current code | Recommended change |
|---|---|---|
| Next | Discovery keeps all paths, sorts them, and loads a full-library Python metadata dictionary. `_existing_files` also turns any read exception into an empty catalog. | Stream discovery into a durable, source-scoped manifest; fail clearly on catalog read errors. Persist scan checkpoints and last-seen generation; expose progress/cancellation during discovery. |
| Next | Every image-table version change can rebuild the complete browse snapshot, including OCR; OCR searches loop over text in Python. | Measure invalidation under active ingestion; coalesce snapshot refreshes and add an indexed text-search path. |
| Next | Broad filtered ANN search can still over-fetch up to the library size, and its heuristic does not guarantee filtered recall. Vector and scalar indexes are recreated after each non-empty indexing run. | Plan filtered queries using scalar predicates; evaluate recall versus exact ranking, maintain indexes incrementally, and schedule training/compaction based on measured thresholds. |
| Next | Full reclustering computes all-pairs similarities in blocks. Memory is partially bounded but time is still quadratic; the default guard permits 400k faces. | Construct a nearest-neighbor graph with ANN, retain confirmed identities, and support resumable clustering. Do not interpret the 400k guard as a demonstrated capacity. |
| Next | Several service methods cap album and face reads at fixed limits (100k/500k), and the album detail view takes at most 500 images. | Add proper pagination/streaming and make truncation visible instead of silently omitting records. |

**Multiple-drive design**

Keep one catalog, model cache, and thumbnail cache on an always-available local
SSD; allow originals across several disks or network sources. Storage settings
already allow these locations to differ. Sharing a catalog does not require
copying all originals onto one disk.

The implemented “Locate folder” flow handles a drive letter or root-folder move
and keeps stable source/photo IDs. A future storage model could separate a logical
photo from several physical replicas sharing one content hash. Source availability
is shown per root; caching prevents grid requests from requiring the original.

Treat offline, inaccessible, and missing as different states. A source disappearing
is not evidence that its photos were deleted. The pruning changes reduce partial
scan hazards but do not identify swapped volumes or make filesystem scans atomic.
Keep pruning explicit until source identity and scan-generation checks exist.

**The requested search is already mostly present**

The desktop UI selects multiple people and defaults to requiring all of them.
Its search request combines `people_ids`, `people_mode: "all"`, and the typed
query. Selecting two people and entering “portrait at sunset” therefore already
uses identity filtering plus semantic scene ranking. It relies on correct face
assignments. “Best quality + match” now reranks those matches with technical and
selected-face quality.

The current SigLIP 2 model supports image/text retrieval; replacing it with a
generative LLM is unnecessary for this first stage. See the
[SigLIP 2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2).

**Implemented “best photos” feature**

The pipeline uses required people and date/source filters, semantic scene
retrieval, quality reranking, and an explicit burst-comparison view.
Identity membership remains a hard constraint throughout. Keep relevance and
technical quality as separate scores so a sharp unrelated photo cannot win.

Store inexpensive, versioned measurements at ingest or through resumable
backfill: resolution, aspect ratio, shadow/highlight clipping, global sharpness,
and sharpness/size of each selected person's face. Existing face quality combines
detector confidence, crop size, and sharpness; it is an identity-quality heuristic,
not a measure of how attractive or meaningful the photo is. For a two-person
portrait, use the weaker selected face as a signal, so one sharp face cannot hide
the other's severe blur.

Normalize measurements to a consistent working resolution. Tune against a small
user-rated set covering portraits, night photos, action, and intentional shallow
depth of field. Avoid universal blur/exposure rejection: sunset silhouettes and
soft backgrounds may be intentional. Laplacian-based sharpness is sensitive to
noise and texture; [OpenCV's focus-measure comparison](https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/)
describes the tradeoffs. Eye closure, expression, pose, and subject segmentation
need additional models or landmarks; they are not supplied by a blur threshold.

The UI offers “Best match”, “Best quality + match”, burst comparison, favorites,
ratings, and per-result explanations. It keeps originals and stores versioned
quality attributes separately from embeddings so tuning need not re-embed the
collection. Automatic one-per-burst collapsing is still future work; comparison
shows a suggested keeper and lets the user decide.

An optional local vision-language model could caption or assess just the top
20–50 candidates, cache structured observations by content hash/model version,
and rerank on demand. A text-only LLM cannot inspect an uncaptioned image.
[Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) is one
candidate to benchmark for this role, not a selected dependency. Test local
latency, memory, and accuracy first. Keep people recognition in the existing
confirmed identity system; do not ask the VLM to infer who a person is. Running
a generative model over every original at every search would be expensive.

**What would make the product feel complete**

Remaining completion work includes smart albums that refresh automatically,
automatic one-per-burst views, full catalog/model backup, and storage/cache usage
controls. Videos currently use one poster frame, so scene search across the whole
clip would be a later feature with timestamped frame embeddings.

**Validation limits**

The full suite passed: `python -m pytest` reported 273 passed in 580.46 seconds,
with one existing Starlette/httpx deprecation warning. New regression coverage
includes recoverable multi-table edits, safe replacement, monotonic IDs,
verified relocation, exact curation restore, failure retry, deterministic
quality ranking, burst comparison, partial/erroring scans, pruning one source
while another is offline, cached previews offline, combined people/text search,
and exact subset ranking.

A separate temporary real LanceDB catalog contained 200,000 synthetic image rows
with 64-dimensional normalized vectors, 500 folders, 400 people, and an image-ID
BTREE index. Using the production service, browse snapshot construction took
0.957 seconds; page 3000 at 60 photos/page took a median 16.22 ms; cosine ranking
a 500-photo person subset took a median 38.56 ms. Query medians use five repeated
calls. These warm synthetic measurements exclude model inference, ANN training,
OCR, physical originals, and external-drive I/O; production SigLIP vectors are
larger. The temporary catalog was removed after measurement.

The existing scale fixture has 200k synthetic photos. Its tests cover NumPy
filtering, ordering, slicing, and duplicate grouping, not real-drive traversal,
model quality, ANN recall, or end-to-end latency. Measuring its retained arrays,
lists, and dictionaries with recursive `sys.getsizeof` on this Windows/Python
3.11 environment gives approximately 39.95 MiB; it still omits production
longitude, camera, and OCR content, database/native allocations, and model memory.

Before claiming a larger supported size, benchmark real database-backed cold
and warm searches at 200k and 1M images, ingestion/reindexing during browsing,
selective people pairs, long OCR text, large face counts, drive disconnects and
letter changes, and restart during writes. Record p50/p95 latency, peak process
RSS, disk/cache growth, scan throughput, and ANN recall against exact results.
