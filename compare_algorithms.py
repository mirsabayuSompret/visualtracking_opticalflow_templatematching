"""
Optical Flow vs Template Matching – Performance Comparison
==========================================================
Dataset : maxwinkelmann/kite-tracking  (Kaggle – image sequence)

The script:
  1. Downloads the dataset from Kaggle.
  2. Reads frames in sorted filename order.
  3. For each consecutive pair computes:
       • Sparse Optical Flow  (Lucas-Kanade, implemented from scratch)
       • Template Matching    (grid-of-patches + NCC sliding search)
  4. Measures wall-clock time per frame-pair for both methods.
  5. Computes per-frame agreement metrics between the two motion maps.
  6. Visualises everything as heat-maps and saves the figures to ./results/.

Lucas-Kanade implementation notes
----------------------------------
The LK optical-flow algorithm assumes brightness constancy and small motion.
For a pixel (x, y) with local window W it solves the over-determined system

    A · [u, v]^T = b

where each row of A is [Ix, Iy] for a pixel in the window and b[i] = -It[i].
The least-squares solution is [u, v]^T = (A^T A)^{-1} A^T b.

This implementation:
  • Uses only NumPy for all mathematical operations.
  • Uses cv2 only for image-level Gaussian smoothing (image pre-processing).
  • Applies a multi-scale (image pyramid) strategy so larger motions are captured.
"""

import os
import time
import warnings
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe in any environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────── configuration ────────────────────────────────
KAGGLE_DATASET = "maxwinkelmann/kite-tracking"   # <owner>/<dataset-slug>
KAGGLE_DIR     = Path(r"D:\.cache\kaggle\kite-tracking")  # local download dir
MAX_PAIRS      = None        # None → process every frame in the dataset
RESULT_EVERY   = 10          # save visualisations for every Nth pair
GRID_ROWS      = 8           # template-matching grid rows
GRID_COLS      = 8           # template-matching grid cols
SEARCH_MARGIN  = 20          # pixels around each patch to search
RESULTS_DIR    = Path("results")
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def pil_to_gray(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an 8-bit grayscale NumPy array."""
    return np.array(img.convert("L"))


def resize_gray(img: np.ndarray, max_dim: int = 320) -> np.ndarray:
    """Down-scale if either dimension exceeds max_dim (keeps aspect ratio)."""
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  Algorithm 1 – Lucas-Kanade Optical Flow (from scratch, NumPy only)
# ══════════════════════════════════════════════════════════════════════════════

_SOBEL_SMOOTH = np.array([1, 2, 1], dtype=np.float32)
_SOBEL_DIFF   = np.array([1, 0, -1], dtype=np.float32)


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded  = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    shape   = img.shape + kernel.shape
    strides = padded.strides + padded.strides
    patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return (patches * kernel).sum(axis=(-2, -1))


def _separable_convolve(img, k_row, k_col):
    return _convolve2d(_convolve2d(img, k_row.reshape(1, -1)), k_col.reshape(-1, 1))


def _spatial_gradient_x(img): return _separable_convolve(img, _SOBEL_DIFF, _SOBEL_SMOOTH)
def _spatial_gradient_y(img): return _separable_convolve(img, _SOBEL_SMOOTH, _SOBEL_DIFF)


def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def _gaussian_blur(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    k = _gaussian_kernel_1d(sigma, radius)
    return _separable_convolve(img, k, k)


def _lk_flow_single_scale(prev, curr, win_half=7):
    h, w  = prev.shape
    avg   = (prev + curr) * 0.5
    Ix    = _spatial_gradient_x(avg)
    Iy    = _spatial_gradient_y(avg)
    It    = curr.astype(np.float32) - prev.astype(np.float32)
    win   = 2 * win_half + 1
    box_k = np.ones(win, dtype=np.float32)
    def _box(a): return _separable_convolve(a, box_k, box_k)
    sIxx = _box(Ix * Ix); sIyy = _box(Iy * Iy); sIxy = _box(Ix * Iy)
    sIxt = _box(Ix * It); sIyt = _box(Iy * It)
    det  = sIxx * sIyy - sIxy * sIxy
    mask = np.abs(det) > 1e-6
    u = np.zeros((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)
    u[mask] = (-sIxt[mask] * sIyy[mask] + sIyt[mask] * sIxy[mask]) / det[mask]
    v[mask] = (-sIyt[mask] * sIxx[mask] + sIxt[mask] * sIxy[mask]) / det[mask]
    return u, v


def _downsample(img):
    h2, w2 = img.shape[0] // 2, img.shape[1] // 2
    return 0.25 * (img[:h2*2:2, :w2*2:2] + img[1:h2*2:2, :w2*2:2]
                 + img[:h2*2:2, 1:w2*2:2] + img[1:h2*2:2, 1:w2*2:2])


def _upsample(flow, target_h, target_w):
    h, w = flow.shape
    yi = np.clip((np.arange(target_h) / (target_h / h)).astype(int), 0, h - 1)
    xi = np.clip((np.arange(target_w) / (target_w / w)).astype(int), 0, w - 1)
    return flow[np.ix_(yi, xi)] * 2.0


def _warp_frame(frame, u, v):
    h, w = frame.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    src_y = np.clip(yy + v, 0, h - 1); src_x = np.clip(xx + u, 0, w - 1)
    y0 = src_y.astype(int); x0 = src_x.astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1);   x1 = np.clip(x0 + 1, 0, w - 1)
    fy = src_y - y0; fx = src_x - x0
    return (frame[y0,x0]*(1-fy)*(1-fx) + frame[y1,x0]*fy*(1-fx)
          + frame[y0,x1]*(1-fy)*fx     + frame[y1,x1]*fy*fx).astype(np.float32)


LK_LEVELS = 3; LK_WIN_HALF = 7; LK_SIGMA = 1.0


def optical_flow_motion_map(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Multi-scale Lucas-Kanade optical flow → normalised [0,1] motion magnitude."""
    p = _gaussian_blur(prev.astype(np.float32), LK_SIGMA)
    c = _gaussian_blur(curr.astype(np.float32), LK_SIGMA)
    pyr_p = [p]; pyr_c = [c]
    for _ in range(LK_LEVELS - 1):
        pyr_p.append(_downsample(pyr_p[-1])); pyr_c.append(_downsample(pyr_c[-1]))
    h0, w0 = pyr_p[-1].shape
    u = np.zeros((h0, w0), dtype=np.float32)
    v = np.zeros((h0, w0), dtype=np.float32)
    for level in range(LK_LEVELS - 1, -1, -1):
        lp = pyr_p[level]; lc = pyr_c[level]; lh, lw = lp.shape
        if level < LK_LEVELS - 1:
            u = _upsample(u, lh, lw); v = _upsample(v, lh, lw)
        warped_c = _warp_frame(lc, u, v)
        du, dv = _lk_flow_single_scale(lp, warped_c, win_half=LK_WIN_HALF)
        u += du; v += dv
    magnitude = np.sqrt(u**2 + v**2)
    mag_max = magnitude.max()
    if mag_max > 0: magnitude /= mag_max
    return magnitude.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Algorithm 2 – Template Matching
# ══════════════════════════════════════════════════════════════════════════════

def template_matching_motion_map(
    prev: np.ndarray, curr: np.ndarray,
    grid_rows=GRID_ROWS, grid_cols=GRID_COLS, margin=SEARCH_MARGIN,
) -> np.ndarray:
    """Grid-of-patches NCC template matching → normalised [0,1] motion magnitude."""
    h, w = prev.shape[:2]
    motion_map = np.zeros((h, w), dtype=np.float32)
    ph = h // grid_rows; pw = w // grid_cols
    if ph < 4 or pw < 4: return motion_map
    for r in range(grid_rows):
        for c in range(grid_cols):
            y1, y2 = r*ph, (r+1)*ph; x1, x2 = c*pw, (c+1)*pw
            template = prev[y1:y2, x1:x2]
            sy1 = max(0, y1-margin); sy2 = min(h, y2+margin)
            sx1 = max(0, x1-margin); sx2 = min(w, x2+margin)
            search = curr[sy1:sy2, sx1:sx2]
            if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            dy = (sy1 + max_loc[1]) - y1; dx = (sx1 + max_loc[0]) - x1
            motion_map[y1:y2, x1:x2] = np.sqrt(dx**2 + dy**2)
    max_disp = motion_map.max()
    if max_disp > 0: motion_map /= max_disp
    return motion_map


# ══════════════════════Loa════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(of_map: np.ndarray, tm_map: np.ndarray) -> dict:
    if of_map.shape != tm_map.shape:
        tm_map = cv2.resize(tm_map, (of_map.shape[1], of_map.shape[0]))
    mse  = float(np.mean((of_map - tm_map) ** 2))
    mae  = float(np.mean(np.abs(of_map - tm_map)))
    corr = float(np.corrcoef(of_map.ravel(), tm_map.ravel())[0, 1])
    return dict(mse=mse, mae=mae, correlation=corr,
                of_motion_coverage=float((of_map > 0.1).mean()),
                tm_motion_coverage=float((tm_map > 0.1).mean()))


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

CMAP_OF = "hot"; CMAP_TM = "cool"; CMAP_DIFF = "RdBu"


def save_pair_heatmap(prev_frame, curr_frame, of_map, tm_map, pair_idx, out_dir):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    axes[0].imshow(prev_frame, cmap="gray"); axes[0].set_title("Frame N (prev)", fontsize=9)
    axes[1].imshow(curr_frame, cmap="gray"); axes[1].set_title("Frame N+1 (curr)", fontsize=9)
    im2 = axes[2].imshow(of_map, cmap=CMAP_OF, vmin=0, vmax=1)
    axes[2].set_title("Lucas-Kanade Optical Flow\n(motion magnitude)", fontsize=9)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    im3 = axes[3].imshow(tm_map, cmap=CMAP_TM, vmin=0, vmax=1)
    axes[3].set_title("Template Matching\n(displacement)", fontsize=9)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    diff = of_map - tm_map
    im4 = axes[4].imshow(diff, cmap=CMAP_DIFF, vmin=-1, vmax=1)
    axes[4].set_title("Difference\n(OF − TM)", fontsize=9)
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    for ax in axes: ax.axis("off")
    fig.suptitle(f"Frame pair {pair_idx:03d}", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"pair_{pair_idx:03d}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_summary_heatmaps(results_df, mean_of_map, mean_tm_map, out_dir):
    # Average heat-maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(mean_of_map, cmap=CMAP_OF, vmin=0, vmax=1)
    axes[0].set_title("Average Lucas-Kanade Optical Flow\nMotion Magnitude", fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(mean_tm_map, cmap=CMAP_TM, vmin=0, vmax=1)
    axes[1].set_title("Average Template Matching\nMotion Magnitude", fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    diff = mean_of_map - mean_tm_map
    im2 = axes[2].imshow(diff, cmap=CMAP_DIFF, vmin=-1, vmax=1)
    axes[2].set_title("Difference (OF − TM)\nAverage across all pairs", fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / "summary_average_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Metric bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title, color in zip(
        axes,
        ["mse", "mae", "correlation"],
        ["MSE (lower → more similar)", "MAE (lower → more similar)",
         "Pearson correlation\n(higher → more similar)"],
        ["#e07b54", "#5b9bd5", "#70ad47"],
    ):
        val = results_df[metric].mean()
        ax.bar([metric.upper()], [val], color=color, edgecolor="black", width=0.4)
        ax.set_title(title, fontsize=9); ax.set_ylim(0, max(1.05, val * 1.15))
        ax.set_ylabel("Mean value")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.suptitle("Agreement Metrics – Lucas-Kanade Optical Flow vs Template Matching", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_metrics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Timing
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(results_df))
    ax.plot(x, results_df["of_time_ms"], label="Optical Flow",     color="#e07b54", linewidth=1.5)
    ax.plot(x, results_df["tm_time_ms"], label="Template Matching", color="#5b9bd5", linewidth=1.5)
    ax.fill_between(x, results_df["of_time_ms"], alpha=0.15, color="#e07b54")
    ax.fill_between(x, results_df["tm_time_ms"], alpha=0.15, color="#5b9bd5")
    ax.set_xlabel("Frame pair index"); ax.set_ylabel("Processing time (ms)")
    ax.set_title("Processing Time per Frame Pair"); ax.legend()
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_timing.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Correlation line chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, results_df["correlation"], color="#70ad47", linewidth=2, marker="o",
            markersize=4, label="Pearson corr (OF vs TM)")
    ax.axhline(results_df["correlation"].mean(), linestyle="--", color="gray",
               label=f"Mean = {results_df['correlation'].mean():.3f}")
    ax.set_ylim(-0.05, 1.05); ax.set_xlabel("Frame pair index"); ax.set_ylabel("Correlation")
    ax.set_title("Per-Frame-Pair Correlation between Optical Flow and Template Matching")
    ax.legend()
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Motion coverage
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    coverage_data = results_df[["of_motion_coverage", "tm_motion_coverage"]].rename(
        columns={"of_motion_coverage": "Optical Flow", "tm_motion_coverage": "Template Matching"})
    sns.heatmap(coverage_data.values.T, ax=axes[0], cmap="YlOrRd",
                yticklabels=["Optical Flow", "Template Matching"],
                xticklabels=[str(i) for i in range(len(results_df))],
                cbar_kws={"label": "Motion coverage"}, vmin=0, vmax=1)
    axes[0].set_title("Motion Coverage per Frame Pair\n(fraction of pixels > 10 % threshold)", fontsize=9)
    axes[0].set_xlabel("Frame pair index"); axes[0].tick_params(axis="x", labelsize=6, rotation=90)
    means = coverage_data.mean()
    bars = axes[1].bar(means.index, means.values, color=["#e07b54", "#5b9bd5"],
                       edgecolor="black", width=0.4)
    axes[1].bar_label(bars, fmt="%.3f", padding=3); axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Mean motion coverage"); axes[1].set_title("Average Motion Coverage\nper Algorithm")
    for s in ["top", "right"]: axes[1].spines[s].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_motion_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_performance_table(results_df: pd.DataFrame, out_dir: Path) -> None:
    summary = pd.DataFrame({
        "Metric": ["Avg processing time (ms)", "Avg motion coverage",
                   "Avg motion intensity (mean map value)"],
        "Optical Flow": [
            f"{results_df['of_time_ms'].mean():.2f}",
            f"{results_df['of_motion_coverage'].mean():.4f}",
            f"{results_df['of_mean_intensity'].mean():.4f}",
        ],
        "Template Matching": [
            f"{results_df['tm_time_ms'].mean():.2f}",
            f"{results_df['tm_motion_coverage'].mean():.4f}",
            f"{results_df['tm_mean_intensity'].mean():.4f}",
        ],
    })
    agreement = pd.DataFrame({
        "Agreement Metric": ["MSE (↓ better)", "MAE (↓ better)", "Pearson Corr (↑ better)"],
        "Mean Value": [
            f"{results_df['mse'].mean():.4f}",
            f"{results_df['mae'].mean():.4f}",
            f"{results_df['correlation'].mean():.4f}",
        ],
    })
    print("\n" + "═"*58 + "\n  PERFORMANCE SUMMARY\n" + "═"*58)
    print(summary.to_string(index=False))
    print("\n" + "─"*58 + "\n  AGREEMENT BETWEEN ALGORITHMS\n" + "─"*58)
    print(agreement.to_string(index=False))
    print("═"*58 + "\n")
    summary.to_csv(out_dir / "performance_summary.csv", index=False)
    agreement.to_csv(out_dir / "agreement_metrics.csv", index=False)
    results_df.to_csv(out_dir / "per_pair_results.csv", index=False)
    print(f"Tables saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_frames(n_frames=32, height=240, width=320):
    rng = np.random.default_rng(42)
    bg  = cv2.GaussianBlur(rng.integers(30, 80, (height, width), dtype=np.uint8), (31, 31), 0)
    objects = [
        (60, 120, 4, 1, 18, 210, "circle"), (200, 80, -3, 2, 14, 180, "circle"),
        (280, 160, 2, -2, 22, 230, "circle"), (100, 200, -2, -1, 15, 195, "circle"),
        (160, 40, 3, 3, 12, 170, "rect"),    (240, 200, -4, 1, 20, 220, "rect"),
    ]
    frames = []; orig_frames = []
    for t in range(n_frames):
        frame = np.clip(bg.copy().astype(np.float32) + rng.normal(0, 3, (height, width)), 0, 255)
        for cx0, cy0, dx, dy, r, bright, shape in objects:
            cx, cy = int((cx0 + dx*t) % width), int((cy0 + dy*t) % height)
            if shape == "circle":
                cv2.circle(frame, (cx, cy), r, float(bright), -1)
                cv2.circle(frame, (cx, cy), r, float(bright - 30), 2)
            else:
                cv2.rectangle(frame, (max(0, cx-r), max(0, cy-r)),
                              (min(width-1, cx+r), min(height-1, cy+r)), float(bright), -1)
        # Save original (pre-blur) frame as RGB for display
        orig = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        orig_frames.append(orig)
        kernel = np.zeros((5, 5)); kernel[2, :] = 0.2
        frames.append(np.clip(cv2.filter2D(frame, -1, kernel), 0, 255).astype(np.uint8))
    return frames, orig_frames


def _frames_from_video_bytes(video_bytes: bytes, max_frames: int) -> list[np.ndarray]:
    """Placeholder kept for compatibility – not used with Kaggle image datasets."""
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loading  (Kaggle – image sequence)
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _download_kaggle_dataset() -> Path:
    """
    Download the Kaggle dataset to KAGGLE_DIR if not already present.
    Requires the kaggle package and either:
      - ~/.kaggle/kaggle.json  OR
      - KAGGLE_USERNAME + KAGGLE_KEY environment variables
    Returns the directory containing the extracted files.
    """
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse cache if images already present
    img_files = [f for f in KAGGLE_DIR.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS]
    if img_files:
        print(f"  → Using cached dataset in: {KAGGLE_DIR}  ({len(img_files)} image files found)")
        return KAGGLE_DIR

    try:
        import kaggle
    except ImportError:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )

    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET} …")
    print(f"  → Destination: {KAGGLE_DIR}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(KAGGLE_DIR),
        unzip=True,
        quiet=False,
    )
    print("  → Download complete.")
    return KAGGLE_DIR


def _collect_image_files(root: Path) -> list[Path]:
    """
    Recursively collect all image files under root, sorted naturally by
    filename so that frame001.jpg < frame002.jpg < … (temporal order).
    """
    return sorted(
        [f for f in root.rglob("*") if f.suffix.lower() in IMG_EXTENSIONS],
        key=lambda p: (p.parent, p.name),
    )


def load_frame_pairs(max_pairs: int | None = MAX_PAIRS):
    """
    Download (or reuse cached) the Kaggle dataset, collect image files in
    sorted order, and return a list of (prev_gray, curr_gray, curr_orig_rgb)
    triplets ready for algorithm processing.
    """
    data_dir  = _download_kaggle_dataset()
    img_files = _collect_image_files(data_dir)

    if len(img_files) < 2:
        raise RuntimeError(
            f"Fewer than 2 image files found in {data_dir}.\n"
            "Check that the dataset downloaded correctly."
        )

    limit     = (max_pairs + 1) if max_pairs else len(img_files)
    img_files = img_files[:limit]
    print(f"  → Found {len(img_files)} image file(s) – building frame pairs …")

    frames: list[np.ndarray]      = []
    orig_frames: list[np.ndarray] = []

    for path in tqdm(img_files, desc="Loading frames", unit="img"):
        try:
            pil_img  = Image.open(path).convert("RGB")
            orig_rgb = np.array(pil_img)
            gray     = resize_gray(np.array(pil_img.convert("L")))
            # resize orig to match gray spatial size
            orig_rgb = cv2.resize(orig_rgb, (gray.shape[1], gray.shape[0]),
                                  interpolation=cv2.INTER_AREA)
            frames.append(gray)
            orig_frames.append(orig_rgb)
        except Exception as e:
            print(f"  [WARN] Skipping {path.name}: {e}")
            continue

    if len(frames) < 2:
        raise RuntimeError("Fewer than 2 frames could be loaded from the dataset.")

    pairs = [(frames[i], frames[i + 1], orig_frames[i + 1])
             for i in range(len(frames) - 1)]
    print(f"  → {len(pairs)} consecutive frame-pair(s) prepared.")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  Bounding Box Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def draw_motion_bounding_boxes(
    frame: np.ndarray,
    motion_map: np.ndarray,
    threshold: float = 0.2,
    min_area: int = 100,
    color: tuple = (0, 255, 0),
    label: str = "",
) -> np.ndarray:
    """
    Draw bounding boxes around regions of significant motion on top of the original frame.

    Parameters
    ----------
    frame       : uint8 grayscale or BGR image
    motion_map  : float32 normalised [0,1] motion magnitude map
    threshold   : motion values above this are considered active (default 0.2)
    min_area    : minimum contour area in pixels to draw a box (filters noise)
    color       : BGR color for the bounding box rectangle
    label       : algorithm name shown next to each box

    Returns
    -------
    vis : BGR uint8 image with bounding boxes drawn
    """
    # Convert grayscale to BGR so we can draw coloured boxes
    if frame.ndim == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()

    # Resize motion map to match the frame if needed
    if motion_map.shape != frame.shape[:2]:
        motion_map = cv2.resize(motion_map, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    # Threshold → binary mask
    binary = (motion_map > threshold).astype(np.uint8) * 255

    # Morphological operations to merge nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)

    # Find contours and draw bounding boxes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(vis, label, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


def save_bounding_box_visualization(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    curr_orig: np.ndarray,
    of_map: np.ndarray,
    tm_map: np.ndarray,
    pair_idx: int,
    out_dir: Path,
    threshold: float = 0.2,
) -> None:
    """
    Save a 5-panel figure showing:
      • Raw original frame (no preprocessing, no boxes)
      • Preprocessed grayscale frame used by algorithms
      • Preprocessed frame + Optical Flow bounding boxes  (green)
      • Preprocessed frame + Template Matching bounding boxes  (yellow)
      • Preprocessed frame + BOTH sets of boxes combined
    """
    of_vis = draw_motion_bounding_boxes(
        curr_frame, of_map, threshold=threshold, color=(0, 255, 0),   label="OF")
    tm_vis = draw_motion_bounding_boxes(
        curr_frame, tm_map, threshold=threshold, color=(0, 200, 255), label="TM")

    # Combined: TM boxes first, then OF on top
    combined = draw_motion_bounding_boxes(
        curr_frame, tm_map, threshold=threshold, color=(0, 200, 255), label="TM")
    combined = draw_motion_bounding_boxes(
        combined,   of_map, threshold=threshold, color=(0, 255, 0),   label="OF")

    fig, axes = plt.subplots(1, 5, figsize=(30, 5))

    # Panel 0 – raw original (no preprocessing)
    axes[0].imshow(curr_orig)  # always RGB
    axes[0].set_title("Original Frame\n(no preprocessing / segmentation)", fontsize=10)

    # Panel 1 – preprocessed gray (resized, what the algorithms see)
    axes[1].imshow(curr_frame, cmap="gray")
    axes[1].set_title("Preprocessed Frame\n(grayscale + resized)", fontsize=10)

    # Panel 2 – OF boxes on preprocessed
    axes[2].imshow(cv2.cvtColor(of_vis, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Lucas-Kanade Optical Flow\nBounding Boxes (green)", fontsize=10)

    # Panel 3 – TM boxes on preprocessed
    axes[3].imshow(cv2.cvtColor(tm_vis, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Template Matching\nBounding Boxes (yellow)", fontsize=10)

    # Panel 4 – both overlaid
    axes[4].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    axes[4].set_title("Combined\nOF (green)  +  TM (yellow)", fontsize=10)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Motion Bounding Boxes – pair {pair_idx:03d}", fontsize=12, y=1.01)
    plt.tight_layout()

    bbox_dir = out_dir / "bounding_boxes"
    bbox_dir.mkdir(exist_ok=True)
    fig.savefig(bbox_dir / f"bbox_{pair_idx:03d}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ...existing code...

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs_dir = RESULTS_DIR / "pairs"
    pairs_dir.mkdir(exist_ok=True)

    pairs = load_frame_pairs(MAX_PAIRS)

    records      = []
    of_maps_all  = []
    tm_maps_all  = []

    print(f"\nProcessing {len(pairs)} frame pair(s) …\n")
    for idx, (prev, curr, curr_orig) in enumerate(tqdm(pairs, unit="pair")):

        # ── Optical Flow ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        of_map = optical_flow_motion_map(prev, curr)
        of_time = (time.perf_counter() - t0) * 1000

        # ── Template Matching ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        tm_map = template_matching_motion_map(prev, curr)
        tm_time = (time.perf_counter() - t0) * 1000

        # resize tm_map to of_map shape (same frame, should already match)
        if tm_map.shape != of_map.shape:
            tm_map = cv2.resize(tm_map, (of_map.shape[1], of_map.shape[0]))

        # ── Metrics ───────────────────────────────────────────────────────────
        m = compute_metrics(of_map, tm_map)
        records.append(dict(
            pair_idx=idx,
            of_time_ms=of_time,
            tm_time_ms=tm_time,
            of_mean_intensity=float(of_map.mean()),
            tm_mean_intensity=float(tm_map.mean()),
            **m,
        ))

        of_maps_all.append(of_map)
        tm_maps_all.append(tm_map)

        # ── Per-pair heat-map figure (every RESULT_EVERY pairs) ──────────────
        if idx % RESULT_EVERY == 0:
            save_pair_heatmap(prev, curr, of_map, tm_map, idx, pairs_dir)

            # ── Bounding box visualisation ────────────────────────────────────
            save_bounding_box_visualization(prev, curr, curr_orig, of_map, tm_map, idx, RESULTS_DIR)

    results_df = pd.DataFrame(records)

    # ── Average motion maps (same spatial resolution as last processed pair) ──
    target_shape = of_maps_all[-1].shape
    of_stack = np.stack([
        cv2.resize(m, (target_shape[1], target_shape[0])) for m in of_maps_all
    ])
    tm_stack = np.stack([
        cv2.resize(m, (target_shape[1], target_shape[0])) for m in tm_maps_all
    ])
    mean_of_map = of_stack.mean(axis=0)
    mean_tm_map = tm_stack.mean(axis=0)

    # ── Summary figures ───────────────────────────────────────────────────────
    print("\nGenerating summary visualisations …")
    save_summary_heatmaps(results_df, mean_of_map, mean_tm_map, RESULTS_DIR)
    save_performance_table(results_df, RESULTS_DIR)

    print(f"\nAll results saved to: {RESULTS_DIR.resolve()}/")
    print("Files:")
    for f in sorted(RESULTS_DIR.rglob("*.png")):
        print(f"  {f}")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
