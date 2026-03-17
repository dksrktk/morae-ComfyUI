"""Batch job queue with progress tracking and crash recovery."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from .client import ComfyUIClient, ComfyUIError
from .config import PipelineConfig
from .workflow import WorkflowTemplate

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    template_path: str
    params: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    output_files: list[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        d["status"] = JobStatus(d["status"])
        return cls(**d)


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    jobs: list[Job] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)

    @property
    def failed(self) -> int:
        return sum(1 for j in self.jobs if j.status == JobStatus.FAILED)

    @property
    def pending(self) -> int:
        return sum(1 for j in self.jobs if j.status in (JobStatus.PENDING, JobStatus.RUNNING))

    @property
    def total(self) -> int:
        return len(self.jobs)

    def save(self, path: Path) -> None:
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "jobs": [j.to_dict() for j in self.jobs],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> SessionState:
        data = json.loads(path.read_text())
        state = cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
        )
        state.jobs = [Job.from_dict(j) for j in data["jobs"]]
        return state


class QueueManager:
    """Manages batch job execution with progress tracking and resume capability."""

    def __init__(self, client: ComfyUIClient, config: PipelineConfig):
        self.client = client
        self.config = config
        self.state: Optional[SessionState] = None
        self._on_job_complete = None  # callback(job, output_files)

    def on_job_complete(self, callback):
        """Register a callback for when a job completes."""
        self._on_job_complete = callback

    def create_session(self, session_id: str) -> SessionState:
        self.state = SessionState(session_id=session_id)
        return self.state

    def load_session(self, session_dir: Path) -> SessionState:
        state_file = session_dir / "state.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No session state at {state_file}")
        self.state = SessionState.load(state_file)
        logger.info(
            f"Resumed session '{self.state.session_id}': "
            f"{self.state.completed}/{self.state.total} completed, "
            f"{self.state.failed} failed"
        )
        return self.state

    def add_jobs(self, jobs: list[Job]) -> None:
        if self.state is None:
            raise RuntimeError("No active session. Call create_session() first.")
        self.state.jobs.extend(jobs)

    def _session_dir(self) -> Path:
        return Path(self.config.output_dir) / self.state.session_id

    def _state_file(self) -> Path:
        return self._session_dir() / "state.json"

    def _raw_dir(self) -> Path:
        return self._session_dir() / "raw"

    async def run(self) -> SessionState:
        """Execute all pending jobs sequentially."""
        if self.state is None:
            raise RuntimeError("No active session")

        session_dir = self._session_dir()
        raw_dir = self._raw_dir()
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Save initial state
        self.state.save(self._state_file())

        pending_jobs = [j for j in self.state.jobs if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]
        total = self.state.total
        times: list[float] = []

        for i, job in enumerate(pending_jobs):
            done = self.state.completed
            eta = ""
            if times:
                avg = sum(times) / len(times)
                remaining = (total - done) * avg
                eta = f" | ETA: {remaining / 60:.1f}min"

            logger.info(f"[{done + 1}/{total}] Job {job.job_id}{eta}")

            job.status = JobStatus.RUNNING
            self.state.save(self._state_file())

            start = time.time()
            retries = self.config.queue.retry_on_failure

            for attempt in range(retries + 1):
                try:
                    template = WorkflowTemplate.from_file(Path(job.template_path))

                    # Apply params
                    if "prompt" in job.params:
                        template.set_prompt(
                            job.params["prompt"],
                            job.params.get("negative"),
                        )
                    if "seed" in job.params:
                        template.set_seed(job.params["seed"])
                    if "width" in job.params and "height" in job.params:
                        template.set_dimensions(job.params["width"], job.params["height"])
                    if "filename_prefix" in job.params:
                        template.set_filename_prefix(job.params["filename_prefix"])

                    # Custom node params
                    for key, value in job.params.get("node_overrides", {}).items():
                        node_id, field = key.split(".", 1)
                        template.set_param(node_id, field, value)

                    output_files = await self.client.generate_and_download(
                        template.to_dict(),
                        raw_dir,
                        timeout=self.config.queue.timeout_seconds,
                    )

                    elapsed = time.time() - start
                    job.status = JobStatus.COMPLETED
                    job.output_files = [str(f) for f in output_files]
                    job.duration_seconds = elapsed
                    times.append(elapsed)

                    if self._on_job_complete and output_files:
                        self._on_job_complete(job, output_files)

                    logger.info(f"  Completed in {elapsed:.1f}s -> {len(output_files)} images")
                    break

                except ComfyUIError as e:
                    if attempt < retries:
                        logger.warning(f"  Attempt {attempt + 1} failed: {e}. Retrying...")
                        await asyncio.sleep(2)
                    else:
                        elapsed = time.time() - start
                        job.status = JobStatus.FAILED
                        job.error = str(e)
                        job.duration_seconds = elapsed
                        logger.error(f"  Failed: {e}")

            self.state.save(self._state_file())

        logger.info(
            f"Session '{self.state.session_id}' done: "
            f"{self.state.completed} completed, {self.state.failed} failed"
        )
        return self.state
