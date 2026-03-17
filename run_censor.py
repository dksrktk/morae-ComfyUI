#!/usr/bin/env python3
"""Run censorship pipeline on a directory of images."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

from morae_pipeline.config import CensorshipConfig
from morae_pipeline.censorship.pipeline import CensorshipPipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_censor.py <input_dir> [output_dir]")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./output_censored")

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Collect image files
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in image_exts
    ])

    if not image_paths:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Output directory: {output_dir}")

    # Setup config (use absolute paths for SAM2)
    script_dir = Path(__file__).parent.resolve()
    config = CensorshipConfig(
        enabled=True,
        device="cuda",
        confidence_threshold=0.25,
        anus_confidence_threshold=0.2,
        bbox_expand_ratio=0.15,
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_checkpoint=str(script_dir / "models/sam2/sam2.1_hiera_large.pt"),
        erode_pixels=0,
        dilate_pixels=3,
        blur_boundary=3,
        filter_method="white_bar",
    )

    # Run pipeline
    pipeline = CensorshipPipeline(config)
    results = pipeline.process_batch(image_paths, output_dir)

    # Write report
    report_path = output_dir / "censorship_report.json"
    pipeline.write_report(results, report_path)

    # Summary
    censored = sum(1 for r in results if r.was_censored)
    total_detections = sum(r.detection_count for r in results)
    print(f"\n{'='*50}")
    print(f"Complete: {len(results)} images processed")
    print(f"Censored: {censored} images ({total_detections} detections)")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
