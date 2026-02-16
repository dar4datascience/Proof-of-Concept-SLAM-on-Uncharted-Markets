import argparse
import subprocess
import sys
from pathlib import Path


BASE_DEFAULTS = {
    "target_fps": 12.0,
    "max_width": 960,
    "crop": "none",
    "rotation": "none",
    "stabilize": False,
    "stabilize_alpha": 0.9,
    "clahe": False,
    "denoise": False,
    "sharpen": False,
    "slam_skip": 2,
    "slam_max_frames": 1200,
    "slam_max_features": 2500,
    "slam_min_matches": 80,
    "slam_ratio_test": 0.75,
    "slam_translation_scale": 0.25,
}

PROFILE_PRESETS = {
    "fast": {
        "target_fps": 8.0,
        "max_width": 720,
        "stabilize": False,
        "clahe": False,
        "denoise": False,
        "sharpen": False,
        "slam_skip": 3,
        "slam_max_frames": 800,
        "slam_max_features": 1800,
        "slam_min_matches": 60,
        "slam_ratio_test": 0.78,
    },
    "balanced": {
        "target_fps": 12.0,
        "max_width": 960,
        "stabilize": True,
        "clahe": True,
        "denoise": False,
        "sharpen": False,
        "slam_skip": 2,
        "slam_max_frames": 1200,
        "slam_max_features": 2500,
        "slam_min_matches": 80,
        "slam_ratio_test": 0.75,
    },
    "quality": {
        "target_fps": 15.0,
        "max_width": 1280,
        "stabilize": True,
        "clahe": True,
        "denoise": True,
        "sharpen": False,
        "slam_skip": 1,
        "slam_max_frames": 1800,
        "slam_max_features": 3500,
        "slam_min_matches": 100,
        "slam_ratio_test": 0.72,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: preprocess video then run SLAM PoC"
    )

    parser.add_argument("--input", required=True, help="Path to raw source video")
    parser.add_argument(
        "--output_root",
        default="outputs/pipeline_run",
        help="Root folder for pipeline outputs",
    )
    parser.add_argument(
        "--preprocessed_video_name",
        default="preprocessed.mp4",
        help="Output filename for the preprocessed video",
    )
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Preset profile for preprocessing + SLAM defaults",
    )

    # Preprocessor options
    parser.add_argument("--target_fps", type=float, default=None)
    parser.add_argument("--max_width", type=int, default=None)
    parser.add_argument(
        "--crop",
        choices=["none", "center_square", "center_16_9", "center_9_16"],
        default=None,
    )
    parser.add_argument(
        "--rotation",
        choices=["none", "cw90", "ccw90", "180"],
        default=None,
    )
    parser.add_argument("--stabilize", dest="stabilize", action="store_true", default=None)
    parser.add_argument("--no_stabilize", dest="stabilize", action="store_false")
    parser.add_argument("--stabilize_alpha", type=float, default=None)
    parser.add_argument("--clahe", dest="clahe", action="store_true", default=None)
    parser.add_argument("--no_clahe", dest="clahe", action="store_false")
    parser.add_argument("--denoise", dest="denoise", action="store_true", default=None)
    parser.add_argument("--no_denoise", dest="denoise", action="store_false")
    parser.add_argument("--sharpen", dest="sharpen", action="store_true", default=None)
    parser.add_argument("--no_sharpen", dest="sharpen", action="store_false")
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--end_sec", type=float, default=None)

    # SLAM options
    parser.add_argument("--slam_skip", type=int, default=None)
    parser.add_argument("--slam_max_frames", type=int, default=None)
    parser.add_argument("--slam_max_features", type=int, default=None)
    parser.add_argument("--slam_min_matches", type=int, default=None)
    parser.add_argument("--slam_ratio_test", type=float, default=None)
    parser.add_argument("--slam_focal_px", type=float, default=None)
    parser.add_argument("--slam_cx", type=float, default=None)
    parser.add_argument("--slam_cy", type=float, default=None)
    parser.add_argument("--slam_translation_scale", type=float, default=None)

    return parser.parse_args()


def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    profile_values = PROFILE_PRESETS.get(args.profile, {})
    for key, base_default in BASE_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, profile_values.get(key, base_default))
    return args


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_preprocess_cmd(args: argparse.Namespace, preprocessed_video: Path, repo_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "video_preprocessor.py"),
        "--input",
        args.input,
        "--output",
        str(preprocessed_video),
        "--target_fps",
        str(args.target_fps),
        "--max_width",
        str(args.max_width),
        "--crop",
        args.crop,
        "--rotation",
        args.rotation,
        "--stabilize_alpha",
        str(args.stabilize_alpha),
        "--start_sec",
        str(args.start_sec),
    ]

    if args.end_sec is not None:
        cmd.extend(["--end_sec", str(args.end_sec)])
    if args.stabilize:
        cmd.append("--stabilize")
    if args.clahe:
        cmd.append("--clahe")
    if args.denoise:
        cmd.append("--denoise")
    if args.sharpen:
        cmd.append("--sharpen")

    return cmd


def build_slam_cmd(args: argparse.Namespace, preprocessed_video: Path, slam_output_dir: Path, repo_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "slam_poc.py"),
        "--video",
        str(preprocessed_video),
        "--output_dir",
        str(slam_output_dir),
        "--skip",
        str(args.slam_skip),
        "--max_frames",
        str(args.slam_max_frames),
        "--max_features",
        str(args.slam_max_features),
        "--min_matches",
        str(args.slam_min_matches),
        "--ratio_test",
        str(args.slam_ratio_test),
        "--translation_scale",
        str(args.slam_translation_scale),
    ]

    if args.slam_focal_px is not None:
        cmd.extend(["--focal_px", str(args.slam_focal_px)])
    if args.slam_cx is not None:
        cmd.extend(["--cx", str(args.slam_cx)])
    if args.slam_cy is not None:
        cmd.extend(["--cy", str(args.slam_cy)])

    return cmd


def main() -> None:
    args = apply_profile_defaults(parse_args())
    repo_root = Path(__file__).resolve().parent

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    preprocessed_video = output_root / args.preprocessed_video_name
    slam_output_dir = output_root / "slam_outputs"
    slam_output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cmd = build_preprocess_cmd(args, preprocessed_video, repo_root)
    slam_cmd = build_slam_cmd(args, preprocessed_video, slam_output_dir, repo_root)

    print("Running preprocessing step...")
    run_command(preprocess_cmd)

    print("\nRunning SLAM step...")
    run_command(slam_cmd)

    print("\nPipeline complete.")
    print(f"Preprocessed video: {preprocessed_video.resolve()}")
    print(f"SLAM outputs: {slam_output_dir.resolve()}")


if __name__ == "__main__":
    main()
