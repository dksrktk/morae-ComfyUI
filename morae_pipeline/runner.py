"""Top-level pipeline runner: generate -> curate -> organize."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from .client import ComfyUIClient
from .config import PipelineConfig
from .censorship.pipeline import CensorshipPipeline
from .curation.pipeline import CurationPipeline
from .queue_manager import QueueManager, Job, JobStatus
from .storage import OutputManager

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the full pipeline: generate images, score them, organize by grade."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def run(
        self,
        session_name: str,
        template_path: str,
        param_list: list[dict],
        curate: bool = True,
    ) -> Path:
        """Full pipeline: generate all, then curate on GPU.

        Args:
            session_name: Unique session identifier
            template_path: Path to API-format workflow JSON
            param_list: List of param dicts, one per image to generate
            curate: Whether to run curation after generation

        Returns:
            Path to session directory
        """
        session_dir = Path(self.config.output_dir) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== Morae Pipeline: {session_name} ===")
        logger.info(f"Template: {template_path}")
        logger.info(f"Jobs: {len(param_list)}")
        logger.info(f"Output: {session_dir}")

        # Phase 1: Generate
        start = time.time()
        async with ComfyUIClient(self.config.comfyui) as client:
            qm = QueueManager(client, self.config)
            qm.create_session(session_name)

            jobs = [
                Job(
                    job_id=f"{session_name}_{i:04d}",
                    template_path=template_path,
                    params=params,
                )
                for i, params in enumerate(param_list)
            ]
            qm.add_jobs(jobs)

            state = await qm.run()

        gen_time = time.time() - start
        logger.info(f"Generation complete: {state.completed}/{state.total} in {gen_time / 60:.1f}min")

        # Phase 2: Curate
        if curate and self.config.curation.enabled:
            await self.run_curation(session_dir)

        # Phase 3: Censorship (post-processing)
        if self.config.censorship.enabled:
            self.run_censorship(session_dir)

        return session_dir

    async def resume(self, session_name: str, curate: bool = True) -> Path:
        """Resume a previously interrupted session."""
        session_dir = Path(self.config.output_dir) / session_name

        logger.info(f"=== Resuming: {session_name} ===")

        async with ComfyUIClient(self.config.comfyui) as client:
            qm = QueueManager(client, self.config)
            qm.load_session(session_dir)

            state = await qm.run()

        logger.info(f"Resume complete: {state.completed}/{state.total}")

        if curate and self.config.curation.enabled:
            await self.run_curation(session_dir)

        # Phase 3: Censorship (post-processing)
        if self.config.censorship.enabled:
            self.run_censorship(session_dir)

        return session_dir

    async def run_curation(self, session_dir: Path) -> None:
        """Run curation on all images in session's raw folder."""
        raw_dir = session_dir / "raw"
        if not raw_dir.exists():
            logger.warning(f"No raw directory at {raw_dir}")
            return

        image_paths = sorted(
            p for p in raw_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        )

        if not image_paths:
            logger.warning("No images found to curate")
            return

        logger.info(f"=== Curation: {len(image_paths)} images ===")

        start = time.time()
        pipeline = CurationPipeline(self.config.curation)
        scores = pipeline.score_batch(image_paths)

        # Organize
        storage = OutputManager(session_dir)
        counts = storage.organize(scores)
        storage.write_scores(scores)
        storage.write_summary(scores, counts)

        elapsed = time.time() - start
        logger.info(f"Curation done in {elapsed:.1f}s")
        logger.info(f"A-grade: {counts['A']} images -> {session_dir / 'graded' / 'A'}")

    def run_censorship(self, session_dir: Path) -> None:
        """Run post-processing censorship on graded images.

        Produces dual output:
          - master/  : uncensored originals (moved from graded/)
          - service/ : censored versions for compliant distribution
        """
        # Collect images from graded folders (A, B, C) — skip rejected
        graded_dir = session_dir / "graded"
        if not graded_dir.exists():
            # Fallback to raw/ if curation was skipped
            source_dir = session_dir / "raw"
            if not source_dir.exists():
                logger.warning(f"No images found for censorship in {session_dir}")
                return
            image_paths = sorted(
                p for p in source_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
        else:
            image_paths = []
            for grade in ("A", "B", "C"):
                grade_dir = graded_dir / grade
                if grade_dir.exists():
                    image_paths.extend(sorted(
                        p for p in grade_dir.iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                    ))

        if not image_paths:
            logger.warning("No images found for censorship")
            return

        logger.info(f"=== Censorship: {len(image_paths)} images ===")

        # Master DB: originals stay in graded/ (or raw/)
        # Service DB: censored versions go to service/
        service_dir = session_dir / "service"
        service_dir.mkdir(parents=True, exist_ok=True)

        pipeline = CensorshipPipeline(self.config.censorship)
        results = pipeline.process_batch(image_paths, service_dir)

        if results:
            pipeline.write_report(results, session_dir / "censorship_report.json")

        censored = sum(1 for r in results if r.was_censored)
        logger.info(
            f"Censorship complete: {censored}/{len(results)} images censored -> {service_dir}"
        )
