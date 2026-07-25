"""Background job runner for long operations (indexing, reclustering).

Indexing 200k photos takes hours. The API cannot block on that, and the old
frontend's "load dataset" button called an endpoint that did not exist, so
there was no way to watch progress at all. This gives every long operation an
id, a live progress record, and a cancel switch.

Jobs run in-process on a single worker thread — the heavy lifting is already
parallel inside the indexer, and one library must not be written by two
ingest runs at once.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

PENDING, RUNNING, DONE, FAILED, CANCELLED = (
    "pending", "running", "done", "failed", "cancelled")


class JobCancelled(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    kind: str
    status: str = PENDING
    phase: str = ""
    current: int = 0
    total: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def percent(self) -> float:
        return round(100.0 * self.current / self.total, 1) if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["percent"] = self.percent
        d["elapsed"] = round(
            (self.finished_at or time.time()) - (self.started_at or self.created_at), 1)
        return d


class JobManager:
    """Serialised background jobs with progress reporting."""

    def __init__(self, state_dir: Optional[str] = None, history: int = 50):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._cancel: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._history = history
        self._state_dir = Path(state_dir) if state_dir else None
        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)

    # -- submission ------------------------------------------------------
    def submit(self, kind: str, fn: Callable[..., Dict[str, Any]], **kwargs) -> Job:
        """Queue a job. Raises if one of the same kind is already active."""
        with self._lock:
            active = self.active()
            if active is not None:
                raise RuntimeError(
                    f"A {active.kind} job is already running (id={active.id}). "
                    "Wait for it to finish or cancel it first.")

            job = Job(id=uuid.uuid4().hex[:12], kind=kind)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._cancel[job.id] = threading.Event()
            self._trim()

        thread = threading.Thread(
            target=self._run, args=(job, fn, kwargs), name=f"job-{kind}", daemon=True)
        self._worker = thread
        thread.start()
        return job

    def _run(self, job: Job, fn: Callable[..., Dict[str, Any]], kwargs: dict) -> None:
        job.status = RUNNING
        job.started_at = time.time()
        cancel = self._cancel[job.id]

        def progress(phase: str, current: int, total: int, detail: dict) -> None:
            if cancel.is_set():
                raise JobCancelled()
            job.phase, job.current, job.total = phase, current, total
            if detail:
                job.detail = detail

        try:
            job.result = fn(progress=progress, **kwargs)
            job.status = DONE
        except JobCancelled:
            job.status = CANCELLED
            job.error = "Cancelled by request"
        except Exception as exc:
            job.status = FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            logger.error("Job %s (%s) failed:\n%s", job.id, job.kind,
                         traceback.format_exc())
        finally:
            job.finished_at = time.time()
            self._persist(job)

    # -- queries ---------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order) if i in self._jobs]

    def active(self) -> Optional[Job]:
        for job in self._jobs.values():
            if job.status in (PENDING, RUNNING):
                return job
        return None

    def cancel(self, job_id: str) -> bool:
        event = self._cancel.get(job_id)
        job = self._jobs.get(job_id)
        if not event or not job or job.status not in (PENDING, RUNNING):
            return False
        event.set()
        return True

    # -- housekeeping ----------------------------------------------------
    def _trim(self) -> None:
        while len(self._order) > self._history:
            old = self._order.pop(0)
            self._jobs.pop(old, None)
            self._cancel.pop(old, None)

    def _persist(self, job: Job) -> None:
        if not self._state_dir:
            return
        try:
            (self._state_dir / f"job-{job.id}.json").write_text(
                json.dumps(job.to_dict(), indent=2, default=str))
        except OSError as exc:  # pragma: no cover
            logger.debug("Could not persist job %s: %s", job.id, exc)
