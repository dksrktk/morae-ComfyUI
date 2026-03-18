"""CLI entry point for Morae Pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .runner import PipelineRunner


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def build_param_list(args) -> list[dict]:
    """Build parameter list from CLI arguments."""
    params = []
    seed = args.seed_start

    for i in range(args.count):
        p = {}
        if args.prompt:
            p["prompt"] = args.prompt
        if args.negative:
            p["negative"] = args.negative
        if args.width and args.height:
            p["width"] = args.width
            p["height"] = args.height

        p["seed"] = seed + i
        p["filename_prefix"] = f"{args.session}_{i:04d}"
        params.append(p)

    return params


def cmd_generate(args) -> None:
    """Generate images in batch."""
    config = _load_config(args)
    runner = PipelineRunner(config)

    if args.resume:
        asyncio.run(runner.resume(args.session, curate=not args.no_curate))
    else:
        if not args.prompt:
            print("Error: --prompt is required for new generation", file=sys.stderr)
            sys.exit(1)
        if not args.template:
            print("Error: --template is required for new generation", file=sys.stderr)
            sys.exit(1)

        param_list = build_param_list(args)
        asyncio.run(
            runner.run(
                session_name=args.session,
                template_path=args.template,
                param_list=param_list,
                curate=not args.no_curate,
            )
        )


def cmd_curate(args) -> None:
    """Re-run curation on existing session."""
    config = _load_config(args)

    # Override thresholds if specified
    if args.threshold_a is not None:
        config.curation.grade_a_min = args.threshold_a
    if args.threshold_b is not None:
        config.curation.grade_b_min = args.threshold_b
    if args.threshold_c is not None:
        config.curation.grade_c_min = args.threshold_c

    runner = PipelineRunner(config)
    session_dir = Path(config.output_dir) / args.session
    if not session_dir.exists():
        print(f"Error: Session directory not found: {session_dir}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(runner.run_curation(session_dir))


def cmd_autopilot(args) -> None:
    """자연어 → 프롬프트 → 배치 생성 → 큐레이션 → 검열 전체 자동화."""
    config = _load_config(args)

    # CLI에서 API 키 오버라이드
    if args.api_key:
        config.generation.deepseek_api_key = args.api_key

    # 트리거 워드 오버라이드
    if args.trigger:
        config.generation.trigger_word = args.trigger

    # Hires 설정
    if args.hires:
        config.generation.hires = True

    runner = PipelineRunner(config)
    asyncio.run(
        runner.autopilot(
            session_name=args.session,
            description=args.prompt,
            count=args.count,
            curate=not args.no_curate,
            censor=not args.no_censor,
            seed_start=args.seed_start,
        )
    )


def cmd_character_batch(args) -> None:
    """캐릭터 + 포즈 기반 배치 생성."""
    config = _load_config(args)

    # CLI에서 API 키 오버라이드
    if args.api_key:
        config.generation.deepseek_api_key = args.api_key

    # 트리거 워드 오버라이드
    if args.trigger:
        config.generation.trigger_word = args.trigger

    # 캐릭터 디렉토리 오버라이드
    if args.characters_dir:
        config.character.characters_dir = args.characters_dir

    # 캐릭터 목록 파싱
    character_names = [c.strip() for c in args.characters.split(",")]

    runner = PipelineRunner(config)
    asyncio.run(
        runner.character_batch(
            session_name=args.session,
            character_names=character_names,
            pose_list_path=args.pose_list,
            auto_poses=args.auto_poses or 0,
            pose_style=args.pose_style or "일반",
            images_per_pose=args.images_per_pose,
            curate=not args.no_curate,
            censor=not args.no_censor,
            seed_start=args.seed_start,
            use_lora=args.use_lora,
        )
    )


def cmd_status(args) -> None:
    """Show session status."""
    import json

    config = _load_config(args)
    output_dir = Path(config.output_dir)

    if args.session:
        sessions = [args.session]
    else:
        if not output_dir.exists():
            print("No sessions found.")
            return
        sessions = [d.name for d in output_dir.iterdir() if d.is_dir()]

    for name in sorted(sessions):
        state_file = output_dir / name / "state.json"
        if not state_file.exists():
            print(f"  {name}: no state file")
            continue

        data = json.loads(state_file.read_text())
        jobs = data.get("jobs", [])
        completed = sum(1 for j in jobs if j["status"] == "completed")
        failed = sum(1 for j in jobs if j["status"] == "failed")
        total = len(jobs)

        # Check graded counts
        graded_dir = output_dir / name / "graded"
        grade_info = ""
        if graded_dir.exists():
            for grade in ["A", "B", "C", "rejected"]:
                gdir = graded_dir / grade
                if gdir.exists():
                    count = len(list(gdir.glob("*")))
                    grade_info += f" {grade}={count}"

        status = "done" if completed == total else "in-progress"
        print(f"  {name}: [{status}] {completed}/{total} completed, {failed} failed{grade_info}")


def _load_config(args) -> PipelineConfig:
    if hasattr(args, "config") and args.config:
        return PipelineConfig.from_yaml(Path(args.config))
    return PipelineConfig.default()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="morae",
        description="Morae Pipeline — Bulk image generation & auto-curation for ComfyUI",
    )
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate images in batch")
    gen.add_argument("--session", "-s", required=True, help="Session name")
    gen.add_argument("--template", "-t", help="Path to API-format workflow JSON")
    gen.add_argument("--prompt", "-p", help="Positive prompt text")
    gen.add_argument("--negative", "-n", help="Negative prompt text")
    gen.add_argument("--count", type=int, default=10, help="Number of images (default: 10)")
    gen.add_argument("--seed-start", type=int, default=1, help="Starting seed (default: 1)")
    gen.add_argument("--width", type=int, help="Image width")
    gen.add_argument("--height", type=int, help="Image height")
    gen.add_argument("--no-curate", action="store_true", help="Skip curation")
    gen.add_argument("--resume", action="store_true", help="Resume interrupted session")
    gen.set_defaults(func=cmd_generate)

    # curate
    cur = sub.add_parser("curate", help="Re-run curation on existing session")
    cur.add_argument("--session", "-s", required=True, help="Session name")
    cur.add_argument("--threshold-a", type=float, help="Grade A minimum (default: 0.8)")
    cur.add_argument("--threshold-b", type=float, help="Grade B minimum (default: 0.6)")
    cur.add_argument("--threshold-c", type=float, help="Grade C minimum (default: 0.4)")
    cur.set_defaults(func=cmd_curate)

    # status
    st = sub.add_parser("status", help="Show session status")
    st.add_argument("--session", "-s", help="Specific session (omit for all)")
    st.set_defaults(func=cmd_status)

    # autopilot
    ap = sub.add_parser("autopilot", help="자연어 → 프롬프트 → 배치 생성 → 큐레이션 → 검열 전체 자동화")
    ap.add_argument("--session", "-s", required=True, help="세션 이름")
    ap.add_argument("--prompt", "-p", required=True, help="자연어 이미지 설명")
    ap.add_argument("--count", "-n", type=int, default=10, help="생성할 이미지 개수 (기본: 10)")
    ap.add_argument("--seed-start", type=int, default=1, help="시작 시드 (기본: 1)")
    ap.add_argument("--api-key", help="DeepSeek API 키 (또는 config에 설정)")
    ap.add_argument("--trigger", "-t", help="LoRA 트리거 워드 (기본: wakitan)")
    ap.add_argument("--hires", action="store_true", help="Hires Fix 사용")
    ap.add_argument("--no-curate", action="store_true", help="큐레이션 건너뛰기")
    ap.add_argument("--no-censor", action="store_true", help="검열 건너뛰기")
    ap.set_defaults(func=cmd_autopilot)

    # character-batch
    cb = sub.add_parser("character-batch", help="캐릭터 + 포즈 기반 배치 생성")
    cb.add_argument("--session", "-s", required=True, help="세션 이름")
    cb.add_argument("--characters", "-c", required=True, help="캐릭터 이름 (쉼표 구분)")
    cb.add_argument("--characters-dir", help="캐릭터 디렉토리 (또는 config에 설정)")
    cb.add_argument("--pose-list", "-p", help="포즈 목록 YAML 경로 (모드 A)")
    cb.add_argument("--auto-poses", type=int, help="LLM 자동 생성 포즈 개수 (모드 B)")
    cb.add_argument("--pose-style", default="일반", help="자동 생성 시 스타일 (기본: 일반)")
    cb.add_argument("--images-per-pose", "-n", type=int, default=1, help="포즈당 이미지 개수 (기본: 1)")
    cb.add_argument("--seed-start", type=int, default=1, help="시작 시드 (기본: 1)")
    cb.add_argument("--api-key", help="DeepSeek API 키")
    cb.add_argument("--trigger", "-t", help="LoRA 트리거 워드")
    cb.add_argument("--use-lora", action="store_true", help="캐릭터 LoRA 사용 (명시적 요청 시에만)")
    cb.add_argument("--no-curate", action="store_true", help="큐레이션 건너뛰기")
    cb.add_argument("--no-censor", action="store_true", help="검열 건너뛰기")
    cb.set_defaults(func=cmd_character_batch)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
