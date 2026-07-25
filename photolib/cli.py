"""Command-line interface: ``python -m photolib.cli <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import get_settings
from .db import Library
from .service import PhotoService


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty at DEBUG and never say anything useful here.
    for noisy in ("urllib3", "PIL", "matplotlib", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_index(args) -> int:
    from tqdm import tqdm

    from .indexer import Indexer

    settings = get_settings()
    settings.ensure_dirs()
    library = Library(settings.db_uri)

    bar = {"pbar": None}

    def progress(phase, current, total, detail):
        if phase == "ingesting":
            if bar["pbar"] is None:
                bar["pbar"] = tqdm(total=total, desc="Indexing", unit="img")
            bar["pbar"].n = current
            bar["pbar"].set_postfix(faces=detail.get("faces", 0), refresh=False)
            bar["pbar"].refresh()
        elif phase == "scanning":
            print(f"Scanning {detail.get('root', '')} ...", flush=True)

    indexer = Indexer(library, settings, progress=progress)
    stats = indexer.index_directory(
        args.folder, rebuild=args.rebuild, prune_missing=args.prune,
        limit=args.limit)
    if bar["pbar"]:
        bar["pbar"].close()

    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def cmd_search(args) -> int:
    from .browse import Filters

    service = PhotoService(get_settings())
    page = service.search(args.query, Filters(), sort="relevance",
                          page=1, per_page=args.limit)
    for item in page.results:
        score = f"{item.get('score', 0):.3f}" if "score" in item else "  -  "
        print(f"{score}  {item['image_id']:>7}  {item.get('taken_at') or '':<19}  "
              f"{item['filename']}")
    print(f"\n{page.total} matches in {page.took_ms}ms")
    return 0


def cmd_stats(args) -> int:
    service = PhotoService(get_settings())
    print(json.dumps(service.stats(), indent=2, default=str))
    return 0


def cmd_recluster(args) -> int:
    from .faces.cluster import recluster

    settings = get_settings()
    service = PhotoService(settings)
    result = recluster(
        service.library, service.face_backend.dim,
        threshold=args.threshold if args.threshold is not None
        else settings.face_cluster_threshold,
        knn=args.knn or settings.face_cluster_knn,
        min_cluster_size=settings.face_min_cluster_size)
    service._resync_all_image_people()
    print(json.dumps(result, indent=2))
    return 0


def cmd_duplicates(args) -> int:
    service = PhotoService(get_settings())
    groups = service.duplicates(max_distance=args.distance, limit=args.limit)
    print(json.dumps(groups, indent=2))
    print(f"\n{len(groups)} duplicate groups", file=sys.stderr)
    return 0


def cmd_compact(args) -> int:
    settings = get_settings()
    library = Library(settings.db_uri)
    library.compact()
    print(json.dumps(library.build_indexes(settings.ann_min_rows), indent=2))
    return 0


def cmd_openapi(args) -> int:
    """Print the OpenAPI schema, so the committed spec never drifts."""
    from .api.app import create_app

    spec = create_app().openapi()
    if args.format == "json":
        print(json.dumps(spec, indent=2))
        return 0
    try:
        import yaml
    except ImportError:
        print("PyYAML is not installed; falling back to JSON.", file=sys.stderr)
        print(json.dumps(spec, indent=2))
        return 0
    print("# Generated from the FastAPI application — do not edit by hand.")
    print("# Regenerate with:  python -m photolib.cli openapi > openapi.yaml")
    print(yaml.safe_dump(spec, sort_keys=False, width=100), end="")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run("photolib.api.app:app", host=args.host or settings.host,
                port=args.port or settings.port, reload=args.reload,
                workers=1 if args.reload else args.workers)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photolib", description="Local, private photo library")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="Index a folder of photos (incremental)")
    p.add_argument("folder", help="Folder to index, scanned recursively")
    p.add_argument("--rebuild", action="store_true",
                   help="Drop the library and reprocess everything")
    p.add_argument("--prune", action="store_true",
                   help="Remove indexed photos whose files no longer exist")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N files (for a quick trial)")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="Search from the terminal")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="Show library statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("recluster", help="Rebuild all people from face embeddings")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--knn", type=int, default=None)
    p.set_defaults(func=cmd_recluster)

    p = sub.add_parser("duplicates", help="List duplicate / near-duplicate photos")
    p.add_argument("--distance", type=int, default=6)
    p.add_argument("--limit", type=int, default=200)
    p.set_defaults(func=cmd_duplicates)

    p = sub.add_parser("compact", help="Compact the database and rebuild indexes")
    p.set_defaults(func=cmd_compact)

    p = sub.add_parser("openapi", help="Print the OpenAPI schema")
    p.add_argument("--format", choices=["yaml", "json"], default="yaml")
    p.set_defaults(func=cmd_openapi)

    p = sub.add_parser("serve", help="Run the API server")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
