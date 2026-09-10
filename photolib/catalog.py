"""Small durable catalog records and recoverable multi-table mutations.

Lance commits each table atomically. A write-ahead list of table versions lets
us roll back an interrupted multi-table edit when the catalog next opens.
"""
from __future__ import annotations

import json
import os
import functools
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa

RECORDS = "catalog_records"
RECORD_SCHEMA = pa.schema([("key", pa.string()), ("value", pa.string())])


def path_key(path):
    return os.path.normcase(os.path.normpath(str(path)))


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def records(library, prefix):
    if RECORDS not in library.table_names():
        return {}
    table = library.table(RECORDS).to_lance().to_table(
        columns=["key", "value"], filter=f"starts_with(key, {literal(prefix)})")
    return {r["key"][len(prefix):]: json.loads(r["value"])
            for r in table.to_pylist()}


def get(library, key, default=None):
    if RECORDS not in library.table_names():
        return default
    rows = library.table(RECORDS).to_lance().to_table(
        columns=["value"], filter=f"key = {literal(key)}").to_pylist()
    return json.loads(rows[0]["value"]) if rows else default


def selected(library, keys):
    keys = list(keys)
    if not keys or RECORDS not in library.table_names():
        return {}
    predicate = ",".join(literal(k) for k in keys)
    rows = library.table(RECORDS).to_lance().to_table(
        columns=["key", "value"], filter=f"key IN ({predicate})").to_pylist()
    return {r["key"]: json.loads(r["value"]) for r in rows}


def put(library, key, value):
    with library._lock:
        if RECORDS not in library.table_names():
            library.db.create_table(RECORDS, schema=RECORD_SCHEMA)
        data = pa.Table.from_pylist([{"key": key, "value": json.dumps(value, allow_nan=False)}],
                                    schema=RECORD_SCHEMA)
        library.table(RECORDS).merge_insert("key").when_matched_update_all().when_not_matched_insert_all().execute(data)


def delete(library, key):
    with library._lock:
        if RECORDS in library.table_names():
            library.table(RECORDS).delete(f"key = {literal(key)}")


def _recover(library):
    journal = Path(library.uri) / ".pending-edit.json"
    if not journal.exists():
        return
    versions = json.loads(journal.read_text(encoding="utf-8"))
    for name in library.table_names():
        if name not in versions:
            library.db.drop_table(name)
        else:
            library.table(name).restore(versions[name])
    journal.unlink()


@contextmanager
def process_lock(library):
    path = Path(library.uri) / ".edit.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def recover(library):
    with process_lock(library):
        _recover(library)


def serialized(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.library._lock:
            return fn(self, *args, **kwargs)
    return wrapped


@contextmanager
def atomic(library):
    """Recoverable edit. Callers must not drop or compact existing tables."""
    with library._lock:
        if getattr(library, "_editing", False):
            yield
            return
        with process_lock(library):
            _recover(library)
            journal = Path(library.uri) / ".pending-edit.json"
            versions = {n: library.table(n).version for n in library.table_names()}
            with journal.open("x", encoding="utf-8") as out:
                json.dump(versions, out)
                out.flush()
                os.fsync(out.fileno())
            library._editing = True
            try:
                yield
            except BaseException:
                _recover(library)
                raise
            else:
                journal.unlink()
            finally:
                library._editing = False
