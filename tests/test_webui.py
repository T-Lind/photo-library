"""Serving the built UI from the API process.

The packaged desktop app is a single process: the Python backend serves both
``/api/v1`` and the statically-exported web UI. These check that the two
coexist on one origin, that client-side routes resolve, and that the
fallback cannot be used to read files outside the web root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from photolib.webui import find_web_dir


@pytest.fixture
def web_dir(tmp_path: Path) -> Path:
    """A miniature stand-in for `next build` output."""
    root = tmp_path / "web"
    (root / "_next" / "static").mkdir(parents=True)
    (root / "index.html").write_text("<html>shell</html>")
    (root / "people.html").write_text("<html>people</html>")
    (root / "person.html").write_text("<html>person</html>")
    (root / "404.html").write_text("<html>not found</html>")
    (root / "_next" / "static" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("TOP SECRET")
    return root


@pytest.fixture
def web_client(settings, web_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from photolib.api.app import create_app
    from photolib.api.deps import set_service

    monkeypatch.setenv("PHOTO_WEB_DIR", str(web_dir))
    set_service(None)
    with TestClient(create_app(settings)) as client:
        yield client
    set_service(None)


def test_root_serves_the_app_shell(web_client):
    response = web_client.get("/")
    assert response.status_code == 200
    assert "shell" in response.text


@pytest.mark.parametrize("route,expected", [
    ("/people", "people"),
    ("/person", "person"),
])
def test_exported_routes_resolve_without_trailing_slash(web_client, route, expected):
    """`next build --output export` writes people.html, not people/index.html."""
    response = web_client.get(route)
    assert response.status_code == 200
    assert expected in response.text


def test_unknown_route_falls_back_to_the_shell(web_client):
    # /person?id=3 style client-side routes have no file of their own.
    response = web_client.get("/whatever/deep/route")
    assert response.status_code == 200
    assert "shell" in response.text


def test_static_assets_are_served(web_client):
    response = web_client.get("/_next/static/app.js")
    assert response.status_code == 200
    assert "console.log" in response.text


def test_missing_asset_is_a_404_not_the_shell(web_client):
    """An absent .js must fail loudly, not silently return HTML."""
    response = web_client.get("/_next/static/missing.js")
    assert response.status_code == 404
    assert "console.log" not in response.text


def test_api_is_not_shadowed_by_the_static_mount(web_client):
    response = web_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_traversal_cannot_escape_the_web_root(web_client):
    for attack in (
        "/../secret.txt",
        "/_next/../../secret.txt",
        "/%2e%2e/secret.txt",
        "/..%2fsecret.txt",
    ):
        response = web_client.get(attack)
        assert "TOP SECRET" not in response.text, attack


def test_api_only_deployment_still_works(settings, monkeypatch):
    """With no UI built, the backend must serve the API and say so."""
    from fastapi.testclient import TestClient

    from photolib.api.app import create_app
    from photolib.api.deps import set_service

    monkeypatch.setenv("PHOTO_WEB_DIR", "/nonexistent/web")
    set_service(None)
    try:
        with TestClient(create_app(settings)) as client:
            root = client.get("/")
            assert root.status_code == 200
            assert "web_ui" in root.json()
            assert client.get("/api/v1/health").status_code == 200
    finally:
        set_service(None)


def test_find_web_dir_ignores_a_directory_without_an_index(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PHOTO_WEB_DIR", str(empty))
    assert find_web_dir() is None
