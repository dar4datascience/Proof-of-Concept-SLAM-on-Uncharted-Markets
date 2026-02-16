import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monocular visual SLAM-style PoC from a walkthrough video"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory for trajectory/maps",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=2,
        help="Process every Nth frame to reduce compute",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1200,
        help="Maximum processed frames",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=2500,
        help="ORB keypoint budget per frame",
    )
    parser.add_argument(
        "--min_matches",
        type=int,
        default=80,
        help="Minimum matches needed to estimate pose",
    )
    parser.add_argument(
        "--ratio_test",
        type=float,
        default=0.75,
        help="Lowe ratio test threshold",
    )
    parser.add_argument(
        "--focal_px",
        type=float,
        default=None,
        help="Camera focal length in pixels (optional)",
    )
    parser.add_argument(
        "--cx",
        type=float,
        default=None,
        help="Principal point x in pixels (optional)",
    )
    parser.add_argument(
        "--cy",
        type=float,
        default=None,
        help="Principal point y in pixels (optional)",
    )
    parser.add_argument(
        "--translation_scale",
        type=float,
        default=0.25,
        help="Relative translation scale per step (monocular has unknown metric scale)",
    )
    return parser.parse_args()


def build_intrinsics(width: int, height: int, args: argparse.Namespace) -> np.ndarray:
    if args.focal_px is None:
        focal_px = 0.9 * max(width, height)
    else:
        focal_px = args.focal_px

    cx = args.cx if args.cx is not None else width / 2.0
    cy = args.cy if args.cy is not None else height / 2.0

    return np.array(
        [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_frames(video_path: Path, skip: int, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    grabbed = 0
    index = 0

    pbar = tqdm(total=min(total, max_frames * skip), desc="Reading frames")
    while grabbed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if index % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            grabbed += 1

        index += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Not enough frames to estimate motion")

    return frames


def match_descriptors(des1, des2, ratio_test: float):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)
    return good


def triangulate_points(K, R, t, pts1, pts2):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    points4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points3d = (points4d[:3] / points4d[3]).T
    return points3d


def save_trajectory_plot(traj: np.ndarray, out_png: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(traj[:, 0], traj[:, 2], "-", linewidth=2)
    ax.set_title("Estimated Camera Trajectory (X-Z plane)")
    ax.set_xlabel("X (relative units)")
    ax.set_ylabel("Z (relative units)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_topdown_map(points: np.ndarray, out_png: Path, bins: int = 240):
    if len(points) == 0:
        return

    x = points[:, 0]
    z = points[:, 2]

    hist, xedges, zedges = np.histogram2d(x, z, bins=bins)
    hist = np.log1p(hist)

    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(
        hist.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], zedges[0], zedges[-1]],
        aspect="equal",
        cmap="viridis",
    )
    fig.colorbar(img, ax=ax, label="log(point density)")
    ax.set_title("Sparse Top-Down Map from Video")
    ax.set_xlabel("X (relative units)")
    ax.set_ylabel("Z (relative units)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def run_slam_like_poc(args: argparse.Namespace):
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(video_path, skip=args.skip, max_frames=args.max_frames)
    h, w = frames[0].shape
    K = build_intrinsics(w, h, args)

    orb = cv2.ORB_create(nfeatures=args.max_features)

    R_w_c = np.eye(3, dtype=np.float64)
    t_w_c = np.zeros((3, 1), dtype=np.float64)

    trajectory = [t_w_c.ravel().copy()]
    map_points_world = []

    kp_prev, des_prev = orb.detectAndCompute(frames[0], None)

    for i in tqdm(range(1, len(frames)), desc="Estimating poses"):
        kp_curr, des_curr = orb.detectAndCompute(frames[i], None)

        if des_prev is None or des_curr is None:
            kp_prev, des_prev = kp_curr, des_curr
            continue

        matches = match_descriptors(des_prev, des_curr, args.ratio_test)
        if len(matches) < args.min_matches:
            kp_prev, des_prev = kp_curr, des_curr
            trajectory.append(t_w_c.ravel().copy())
            continue

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts_prev,
            pts_curr,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5,
        )
        if E is None:
            kp_prev, des_prev = kp_curr, des_curr
            trajectory.append(t_w_c.ravel().copy())
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts_prev, pts_curr, cameraMatrix=K)

        t = t * float(args.translation_scale)

        R_w_c_new = R_w_c @ R.T
        t_w_c_new = t_w_c - R_w_c_new @ t

        inliers = (mask_pose.ravel() > 0)
        if np.count_nonzero(inliers) > 12:
            pts1_in = pts_prev[inliers]
            pts2_in = pts_curr[inliers]
            pts_cam1 = triangulate_points(K, R, t, pts1_in, pts2_in)
            pts_world = (R_w_c @ pts_cam1.T).T + t_w_c.ravel()

            finite = np.isfinite(pts_world).all(axis=1)
            reasonable = np.linalg.norm(pts_world, axis=1) < 1e3
            keep = finite & reasonable
            if np.any(keep):
                map_points_world.append(pts_world[keep])

        R_w_c, t_w_c = R_w_c_new, t_w_c_new
        trajectory.append(t_w_c.ravel().copy())
        kp_prev, des_prev = kp_curr, des_curr

    traj_arr = np.asarray(trajectory)
    pts_arr = np.vstack(map_points_world) if map_points_world else np.empty((0, 3))

    np.save(output_dir / "trajectory.npy", traj_arr)
    np.save(output_dir / "pointcloud_sparse.npy", pts_arr)
    np.savetxt(output_dir / "trajectory.csv", traj_arr, delimiter=",", header="x,y,z", comments="")

    save_trajectory_plot(traj_arr, output_dir / "trajectory_xz.png")
    save_topdown_map(pts_arr, output_dir / "topdown_density_map.png")

    print(f"Processed frames: {len(frames)}")
    print(f"Trajectory points: {len(traj_arr)}")
    print(f"Sparse map points: {len(pts_arr)}")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    run_slam_like_poc(parse_args())
