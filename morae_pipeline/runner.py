"""Top-level pipeline runner: generate -> curate -> organize."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .client import ComfyUIClient
from .config import PipelineConfig
from .censorship.pipeline import CensorshipPipeline
from .character.profile import CharacterLibrary, CharacterProfile
from .curation.pipeline import CurationPipeline
from .pose_list import PoseList, PoseGenerator
from .prompt_generator import PromptGenerator
from .queue_manager import QueueManager, Job, JobStatus
from .storage import OutputManager
from .workflow import WorkflowBuilder

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

    async def autopilot(
        self,
        session_name: str,
        description: str,
        count: int = 10,
        curate: bool = True,
        censor: bool = True,
        seed_start: int = 1,
    ) -> Path:
        """전체 자동화: 자연어 → 프롬프트 → 배치 생성 → 큐레이션 → 검열.

        Args:
            session_name: 세션 이름
            description: 자연어 이미지 설명
            count: 생성할 이미지 개수
            curate: 큐레이션 실행 여부
            censor: 검열 실행 여부
            seed_start: 시작 시드

        Returns:
            세션 디렉토리 경로
        """
        session_dir = Path(self.config.output_dir) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        gen_config = self.config.generation

        logger.info(f"=== Morae Autopilot: {session_name} ===")
        logger.info(f"Description: {description}")
        logger.info(f"Count: {count}")

        # Step 1: DeepSeek로 프롬프트 생성
        if not gen_config.deepseek_api_key:
            raise ValueError("deepseek_api_key가 설정되지 않았습니다")

        prompt_gen = PromptGenerator(
            api_key=gen_config.deepseek_api_key,
            trigger_word=gen_config.trigger_word,
        )
        generated = prompt_gen.generate(description)

        logger.info(f"[Prompt] {generated.positive[:100]}...")

        # Step 2: 워크플로우 빌더
        builder = WorkflowBuilder(gen_config)

        # Step 3: 배치 생성
        start = time.time()
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        async with ComfyUIClient(self.config.comfyui) as client:
            for i in range(count):
                seed = seed_start + i
                filename_prefix = f"{session_name}_{i:04d}"

                workflow, used_seed = builder.build(
                    positive_prompt=generated.positive,
                    negative_prompt=generated.negative,
                    seed=seed,
                    filename_prefix=filename_prefix,
                )

                logger.info(f"[{i + 1}/{count}] Generating (seed={used_seed})...")

                try:
                    downloaded = await client.generate_and_download(
                        workflow,
                        raw_dir,
                        timeout=self.config.queue.timeout_seconds,
                    )
                    if downloaded:
                        logger.debug(f"  -> {downloaded[0].name}")
                except Exception as e:
                    logger.error(f"  -> Failed: {e}")

        gen_time = time.time() - start
        logger.info(f"Generation complete: {count} images in {gen_time / 60:.1f}min")

        # Step 4: 큐레이션
        if curate and self.config.curation.enabled:
            await self.run_curation(session_dir)

        # Step 5: 검열
        if censor and self.config.censorship.enabled:
            self.run_censorship(session_dir)

        logger.info(f"=== Autopilot Complete: {session_dir} ===")
        return session_dir

    async def character_batch(
        self,
        session_name: str,
        character_names: list[str],
        pose_list_path: Optional[str] = None,
        auto_poses: int = 0,
        pose_style: str = "일반",
        images_per_pose: int = 1,
        curate: bool = True,
        censor: bool = True,
        seed_start: int = 1,
        use_lora: bool = False,
    ) -> Path:
        """캐릭터 + 포즈 기반 배치 생성.

        Args:
            session_name: 세션 이름
            character_names: 캐릭터 이름 목록
            pose_list_path: 포즈 목록 YAML 경로 (모드 A)
            auto_poses: LLM 자동 생성 포즈 개수 (모드 B)
            pose_style: 자동 생성 시 스타일
            images_per_pose: 포즈당 이미지 개수
            curate: 큐레이션 실행 여부
            censor: 검열 실행 여부
            seed_start: 시작 시드

        Returns:
            세션 디렉토리 경로
        """
        session_dir = Path(self.config.output_dir) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        gen_config = self.config.generation
        char_config = self.config.character

        # API 키 확인
        if not gen_config.deepseek_api_key:
            raise ValueError("deepseek_api_key가 설정되지 않았습니다")

        # 캐릭터 라이브러리 로드
        if not char_config.characters_dir:
            raise ValueError("characters_dir이 설정되지 않았습니다")

        char_library = CharacterLibrary(char_config.characters_dir)

        # 캐릭터 검증
        characters: list[CharacterProfile] = []
        for name in character_names:
            char = char_library.get(name)
            if not char:
                raise ValueError(f"캐릭터를 찾을 수 없음: {name}")
            characters.append(char)

        logger.info(f"=== Character Batch: {session_name} ===")
        logger.info(f"Characters: {[c.name for c in characters]}")

        # 포즈 목록 로드/생성
        if pose_list_path:
            pose_list = PoseList.load(Path(pose_list_path))
            logger.info(f"Pose list loaded: {len(pose_list)} poses")
        elif auto_poses > 0:
            pose_gen = PoseGenerator(gen_config.deepseek_api_key)
            pose_list = pose_gen.generate(auto_poses, pose_style)
            # 자동 생성된 포즈 목록 저장
            auto_pose_path = session_dir / "auto_poses.yaml"
            pose_list.save(auto_pose_path)
            logger.info(f"Auto-generated {len(pose_list)} poses -> {auto_pose_path}")
        else:
            raise ValueError("pose_list_path 또는 auto_poses 중 하나를 지정해야 합니다")

        # 프롬프트 생성기
        prompt_gen = PromptGenerator(
            api_key=gen_config.deepseek_api_key,
            trigger_word=gen_config.trigger_word,
        )

        # 워크플로우 빌더
        builder = WorkflowBuilder(gen_config)

        # 배치 생성
        start = time.time()
        seed = seed_start

        async with ComfyUIClient(self.config.comfyui) as client:
            for char in characters:
                char_dir = session_dir / char.name
                char_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"[{char.name}] 생성 시작 ({len(pose_list)} poses x {images_per_pose})")

                # 캐릭터 LoRA 준비 (명시적 요청 시에만)
                char_loras = None
                if use_lora and char.loras:
                    char_loras = [lora.to_dict() for lora in char.loras]
                    logger.info(f"  [LoRA] {len(char_loras)}개 적용: {[l['path'] for l in char_loras]}")

                for pose in pose_list:
                    # LLM으로 캐릭터+포즈 프롬프트 생성
                    generated = prompt_gen.generate_for_character_pose(
                        character_tags=char.tag_string,
                        pose_description=pose.description,
                        character_negative_tags=char.negative_tag_string,
                    )

                    for img_idx in range(images_per_pose):
                        filename_prefix = f"{pose.id}_{img_idx + 1:03d}"

                        workflow, used_seed = builder.build(
                            positive_prompt=generated.positive,
                            negative_prompt=generated.negative,
                            seed=seed,
                            filename_prefix=filename_prefix,
                            character_loras=char_loras,
                        )
                        seed += 1

                        logger.info(f"  [{char.name}/{pose.id}] #{img_idx + 1} (seed={used_seed})")

                        try:
                            downloaded = await client.generate_and_download(
                                workflow,
                                char_dir,
                                timeout=self.config.queue.timeout_seconds,
                            )
                            if downloaded:
                                logger.debug(f"    -> {downloaded[0].name}")
                        except Exception as e:
                            logger.error(f"    -> Failed: {e}")

        gen_time = time.time() - start
        total_images = len(characters) * len(pose_list) * images_per_pose
        logger.info(f"Generation complete: {total_images} images in {gen_time / 60:.1f}min")

        # 큐레이션 (캐릭터별로)
        if curate and self.config.curation.enabled:
            for char in characters:
                char_dir = session_dir / char.name
                # 임시로 raw 디렉토리 생성해서 큐레이션
                raw_dir = char_dir / "raw"
                if not raw_dir.exists():
                    # 이미지들을 raw로 이동
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    for img in char_dir.glob("*.png"):
                        img.rename(raw_dir / img.name)
                    for img in char_dir.glob("*.jpg"):
                        img.rename(raw_dir / img.name)

                await self.run_curation(char_dir)

        # 검열 (캐릭터별로)
        if censor and self.config.censorship.enabled:
            for char in characters:
                self.run_censorship(session_dir / char.name)

        logger.info(f"=== Character Batch Complete: {session_dir} ===")
        return session_dir
