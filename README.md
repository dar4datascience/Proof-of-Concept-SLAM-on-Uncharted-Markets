# Proof-of-Concept-SLAM-on-Uncharted-Markets

This repository contains a **Python proof-of-concept (PoC)** for a SLAM-like pipeline that estimates camera trajectory and builds a sparse top-down map from a walkthrough video.

Use case: test mapping quality for spaces like a **mall** or **warehouse** that do not have an official floor map.

## What this PoC does

- Reads a monocular video (single camera stream)
- Detects and matches ORB features frame-to-frame
- Estimates relative camera motion with epipolar geometry (`findEssentialMat` + `recoverPose`)
- Triangulates sparse 3D points
- Builds:
  - camera trajectory (`trajectory.npy`, `trajectory.csv`, `trajectory_xz.png`)
  - sparse top-down density map (`topdown_density_map.png`)

## Important limitation (monocular SLAM)

Because this is monocular video-only, **absolute metric scale is not observable** without extra signals (known object size, IMU, depth, stereo, wheel odometry, etc.).

So distances are in **relative units**, not true meters.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python slam_poc.py --video /absolute/path/to/video.mp4 --output_dir outputs
```

Useful flags:

- `--skip 2` (default): use every 2nd frame
- `--max_frames 1200`: cap processing for faster experiments
- `--max_features 2500`: ORB feature budget
- `--min_matches 80`: required matches before pose solve
- `--focal_px`, `--cx`, `--cy`: override intrinsics if known
- `--translation_scale 0.25`: visualization scaling for relative translation

Example:

```bash
python slam_poc.py \
  --video /data/mall_walkthrough.mp4 \
  --output_dir outputs/mall_test_01 \
  --skip 3 \
  --max_frames 1000
```

## Input video guidance for better mapping

- Walk smoothly (avoid sudden shakes and rapid rotations)
- Keep overlap high between consecutive views
- Capture textured areas (storefront signs, racks, corners, patterns)
- Avoid long pure-glass / textureless walls when possible
- Keep lighting stable if possible

## Suggested workflow for TikTok videos

1. Obtain the video file legally and in the highest quality available.
2. Preprocess it for SLAM with `video_preprocessor.py`.
3. Run `slam_poc.py` on the preprocessed clip.
4. Inspect `trajectory_xz.png` and `topdown_density_map.png` for structure consistency.
5. Tune preprocessing and SLAM flags together.

## Video preprocessor (recommended before SLAM)

Use this to trim, stabilize, normalize FPS, rotate/crop, resize, and improve contrast.

```bash
python video_preprocessor.py \
  --input /data/raw_tiktok.mp4 \
  --output /data/preprocessed_mall.mp4 \
  --start_sec 5 \
  --end_sec 55 \
  --target_fps 12 \
  --max_width 960 \
  --stabilize \
  --clahe
```

Then run SLAM:

```bash
python slam_poc.py --video /data/preprocessed_mall.mp4 --output_dir outputs/mall_test_01
```

Common preprocessor flags:

- `--crop {none,center_square,center_16_9,center_9_16}`
- `--rotation {none,cw90,ccw90,180}`
- `--stabilize --stabilize_alpha 0.9`
- `--clahe --denoise --sharpen`
- `--start_sec` and `--end_sec` for selecting one continuous route segment

The preprocessor also writes sidecar metadata: `*.metadata.json`.

## One-command pipeline (preprocess + SLAM)

Use `pipeline.py` to run both steps in sequence.

Profiles:

- `--profile fast`: quickest run, lower compute
- `--profile balanced`: default, good first pass
- `--profile quality`: slower, denser processing

You can still override any individual flag after choosing a profile.

```bash
python pipeline.py \
  --input /data/raw_tiktok.mp4 \
  --profile balanced \
  --output_root outputs/mall_pipeline_01 \
  --start_sec 5 \
  --end_sec 55 \
  --target_fps 12 \
  --max_width 960 \
  --stabilize \
  --clahe \
  --slam_skip 2 \
  --slam_max_frames 1200
```

Fast profile example:

```bash
python pipeline.py --input /data/raw_tiktok.mp4 --profile fast --output_root outputs/mall_fast
```

Quality profile example:

```bash
python pipeline.py --input /data/raw_tiktok.mp4 --profile quality --output_root outputs/mall_quality
```

Pipeline outputs:

- `outputs/mall_pipeline_01/preprocessed.mp4`
- `outputs/mall_pipeline_01/preprocessed.metadata.json`
- `outputs/mall_pipeline_01/slam_outputs/*`

## Getting true meters (metric scale)

Monocular video alone cannot recover absolute scale. To get map distances in meters, add at least one scale source:

1. **Stereo camera baseline** (known lens separation) or **RGB-D/depth camera**.
2. **IMU + visual-inertial SLAM (VIO)** with good time synchronization.
3. **Wheel odometry / encoder data** (for wheeled platforms).
4. **Known-size landmarks** in scene (e.g., AprilTags with known tag size).
5. **LiDAR fusion** (2D/3D LiDAR + camera).

Practical note: phone video from social media usually lacks synchronized raw IMU/depth streams, so metric scale is hard unless you also collect sensor logs during capture.

## Next upgrade path

If this PoC shows promise, move to one of these for stronger results:

1. **pySLAM** for a fuller VO/SLAM stack with loop closing options.
2. **COLMAP / PyCOLMAP** for high-quality offline SfM + dense reconstruction.
3. ORB-SLAM/OpenVSLAM-based pipelines when C++ integrations are acceptable.

## Web references used for this PoC direction

- LearnOpenCV: Monocular SLAM in Python/OpenCV
  - https://learnopencv.com/monocular-slam-in-python/
- pySLAM repository (hybrid Python/C++ visual SLAM)
  - https://github.com/luigifreda/pyslam
- COLMAP tutorial (SfM + MVS guidance)
  - https://colmap.github.io/tutorial.html