"""Development server: auto-reload, bound to localhost.

Equivalent to ``python -m photolib.cli serve --reload``.
"""

from __future__ import annotations

import logging

import uvicorn

from photolib.config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = get_settings()
    settings.ensure_dirs()

    print(f"photolib dev server → http://{settings.host}:{settings.port}")
    print(f"  API docs   http://{settings.host}:{settings.port}/docs")
    print(f"  Database   {settings.db_uri}")
    print(f"  Embeddings {settings.embed_backend}:{settings.embed_model}")
    print(f"  Faces      {settings.face_backend}:{settings.face_model}")

    uvicorn.run("photolib.api.app:app", host=settings.host, port=settings.port,
                reload=True, workers=1)


if __name__ == "__main__":
    main()
