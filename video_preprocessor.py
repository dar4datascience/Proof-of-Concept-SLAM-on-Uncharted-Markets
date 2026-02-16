import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess walkthrough videos for SLAM-style pipelines"
    )
    parser.add_argument("--input", required=True, help="Path to source video")
    parser.add_argument("--output", required=True, help="Path to preprocessed video")
    parser.add_argument(
        "--target_fps",
        type=float,
        default=12.0,
        help="Output FPS (downsampled from source). Use <=0 to keep source fps.",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=960,
        help="Resize so output width is at most this many pixels (keeps aspect ratio).",
    )
    parser.add_argument(
        "--crop",
        choices=["none", "center_square", "center_16_9", "center_9_16"],
        default="none",
        help="Center crop mode before resize.",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Apply lightweight motion stabilization.",
    )
    parser.add_argument(
        "--stabilize_alpha",
        type=float,
        default=0.9,
        help="Smoothing factor for stabilization (0.8-0.97 typical).",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE contrast enhancement on luminance channel.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply mild denoising (fastNlMeansDenoisingColored).",
    )
    parser.add_argument(
        "--sharpen",
        action="store_true",
        help="Apply gentle sharpening filter.",
    )
    parser.add_argument(
        "--start_sec",
        type=float,
        default=0.0,
        help="Start time (seconds) in source video.",
    )
    parser.add_argument(
        "--end_sec",
        type=float,
        default=None,
        help="End time (seconds) in source video.",
    )
    parser.add_argument(
        "--rotation",
        choices=["none", "cw90", "ccw90", "180"],
        default="none",
        help="Rotate frames before crop/resize.",
    )
    return parser.parse_args()


def rotate_frame(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cw90":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw90":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def center_crop(frame: np.ndarray, mode: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if mode == "none":
        return frame

    if mode == "center_square":
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return frame[y0 : y0 + side, x0 : x0 + side]

    target_ratio = 16.0 / 9.0 if mode == "center_16_9" else 9.0 / 16.0
    src_ratio = w / float(h)

    if src_ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        return frame[:, x0 : x0 + new_w]

    new_h = int(round(w / target_ratio))
    y0 = (h - new_h) // 2
    return frame[y0 : y0 + new_h, :]


def resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame
    scale = max_width / float(w)
    new_w = max_width
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def enhance_frame(frame: np.ndarray, do_clahe: bool, do_denoise: bool, do_sharpen: bool) -> np.ndarray:
    out = frame

    if do_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    if do_denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)

    if do_sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)

    return out


def estimate_delta_transform(prev_gray: np.ndarray, curr_gray: np.ndarray):
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=15,
        blockSize=3,
    )
    if prev_pts is None or len(prev_pts) < 20:
        return 0.0, 0.0, 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None or status is None:
        return 0.0, 0.0, 0.0

    keep = status.ravel() == 1
    prev_good = prev_pts[keep]
    curr_good = curr_pts[keep]
    if len(prev_good) < 20:
        return 0.0, 0.0, 0.0

    M, _ = cv2.estimateAffinePartial2D(prev_good, curr_good)
    if M is None:
        return 0.0, 0.0, 0.0

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    da = float(np.arctan2(M[1, 0], M[0, 0]))
    return dx, dy, da


def warp_by_transform(frame: np.ndarray, dx: float, dy: float, da: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cos_a = np.cos(da)
    sin_a = np.sin(da)
    M = np.array(
        [[cos_a, -sin_a, dx], [sin_a, cos_a, dy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def preprocess_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_sec = args.end_sec
    if end_sec is None:
        end_sec = total_frames / src_fps if total_frames > 0 else 1e9

    start_frame = max(0, int(round(args.start_sec * src_fps)))
    end_frame = max(start_frame + 1, int(round(end_sec * src_fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    target_fps = src_fps if args.target_fps <= 0 else float(args.target_fps)
    step_time = 1.0 / target_fps

    writer = None
    frame_idx = start_frame
    kept = 0
    elapsed = 0.0

    prev_gray_for_stab = None
    cdx = cdy = cda = 0.0
    sdx = sdy = sda = 0.0

    pbar_total = max(0, min(total_frames, end_frame) - start_frame) if total_frames > 0 else None
    pbar = tqdm(total=pbar_total, desc="Preprocessing video")

    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        elapsed += 1.0 / src_fps
        should_keep = elapsed >= step_time or kept == 0
        if should_keep:
            elapsed = 0.0

            frame = rotate_frame(frame, args.rotation)
            frame = center_crop(frame, args.crop)
            frame = resize_max_width(frame, args.max_width)
            frame = enhance_frame(frame, args.clahe, args.denoise, args.sharpen)

            if args.stabilize:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray_for_stab is not None:
                    dx, dy, da = estimate_delta_transform(prev_gray_for_stab, gray)
                    cdx += dx
                    cdy += dy
                    cda += da

                    alpha = float(np.clip(args.stabilize_alpha, 0.0, 0.999))
                    sdx = alpha * sdx + (1.0 - alpha) * cdx
                    sdy = alpha * sdy + (1.0 - alpha) * cdy
                    sda = alpha * sda + (1.0 - alpha) * cda

                    correction_x = sdx - cdx
                    correction_y = sdy - cdy
                    correction_a = sda - cda
                    frame = warp_by_transform(frame, correction_x, correction_y, correction_a)
                prev_gray_for_stab = gray

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (w, h))

            writer.write(frame)
            kept += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "source_fps": src_fps,
        "target_fps": target_fps,
        "start_sec": args.start_sec,
        "end_sec": None if args.end_sec is None else args.end_sec,
        "frames_written": kept,
        "crop": args.crop,
        "max_width": args.max_width,
        "stabilize": bool(args.stabilize),
        "clahe": bool(args.clahe),
        "denoise": bool(args.denoise),
        "sharpen": bool(args.sharpen),
        "rotation": args.rotation,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote preprocessed video: {output_path.resolve()}")
    print(f"Wrote metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    preprocess_video(parse_args())
