"""
Optical Flow vs Template Matching – Performance Comparison
==========================================================
Dataset : svjack/genshin_impact_slow_motion_splited  (HuggingFace)

The script:
  1. Downloads the dataset from HuggingFace.
  2. Extracts consecutive-frame pairs from every available split.
  3. For each pair computes:
       • Dense Optical Flow  (Farneback)
       • Template Matching   (grid-of-patches + NCC sliding search)
  4. Measures wall-clock time per frame-pair for both methods.
  5. Computes per-frame agreement metrics between the two motion maps.
  6. Visualises everything as heat-maps and saves the figures to ./results/.
"""

import os
import time
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe in any environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────── configuration ────────────────────────────────
DATASET_NAME   = "svjack/genshin_impact_slow_motion_splited"
MAX_PAIRS      = 30          # frame-pairs to process (set None → all)
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
#  Algorithm 1 – Farneback Dense Optical Flow
# ══════════════════════════════════════════════════════════════════════════════

def optical_flow_motion_map(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """
    Returns a normalised (0-1) motion-magnitude map using Farneback optical flow.

    Parameters
    ----------
    prev, curr : uint8 grayscale frames
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # normalise to [0, 1]
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude /= mag_max
    return magnitude.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Algorithm 2 – Template Matching
# ══════════════════════════════════════════════════════════════════════════════

def template_matching_motion_map(
    prev: np.ndarray,
    curr: np.ndarray,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    margin: int = SEARCH_MARGIN,
) -> np.ndarray:
    """
    Divides *prev* into a (grid_rows × grid_cols) grid of non-overlapping
    patches and finds each patch inside *curr* using normalised cross-correlation
    (TM_CCOEFF_NORMED).

    Returns a (H × W) float32 motion-magnitude map normalised to [0, 1]
    where each pixel in a patch carries the displacement magnitude of that patch.
    """
    h, w = prev.shape[:2]
    motion_map = np.zeros((h, w), dtype=np.float32)

    ph = h // grid_rows     # patch height
    pw = w // grid_cols     # patch width

    if ph < 4 or pw < 4:    # frame too small for the chosen grid
        return motion_map

    displacements = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            # ── patch boundaries in *prev* ──
            y1, y2 = r * ph, (r + 1) * ph
            x1, x2 = c * pw, (c + 1) * pw
            template = prev[y1:y2, x1:x2]

            # ── search region in *curr* (with margin) ──
            sy1 = max(0, y1 - margin)
            sy2 = min(h, y2 + margin)
            sx1 = max(0, x1 - margin)
            sx2 = min(w, x2 + margin)
            search = curr[sy1:sy2, sx1:sx2]

            if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                continue

            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            # best-match top-left in *curr* coordinates
            best_y = sy1 + max_loc[1]
            best_x = sx1 + max_loc[0]

            # displacement from original patch position
            dy = best_y - y1
            dx = best_x - x1
            displacement = np.sqrt(dx**2 + dy**2)
            displacements.append(displacement)

            motion_map[y1:y2, x1:x2] = displacement

    # normalise
    max_disp = motion_map.max()
    if max_disp > 0:
        motion_map /= max_disp

    return motion_map


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(of_map: np.ndarray, tm_map: np.ndarray) -> dict:
    """Compute agreement/comparison metrics between two motion maps."""
    # resize tm_map to of_map shape if needed (they should already match)
    if of_map.shape != tm_map.shape:
        tm_map = cv2.resize(tm_map, (of_map.shape[1], of_map.shape[0]))

    mse  = float(np.mean((of_map - tm_map) ** 2))
    mae  = float(np.mean(np.abs(of_map - tm_map)))
    corr = float(np.corrcoef(of_map.ravel(), tm_map.ravel())[0, 1])

    # motion coverage: fraction of pixels above 10 % of max
    of_active = float((of_map > 0.1).mean())
    tm_active = float((tm_map > 0.1).mean())

    return dict(mse=mse, mae=mae, correlation=corr,
                of_motion_coverage=of_active, tm_motion_coverage=tm_active)


# ══════════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

CMAP_OF = "hot"
CMAP_TM = "cool"
CMAP_DIFF = "RdBu"


def save_pair_heatmap(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    of_map: np.ndarray,
    tm_map: np.ndarray,
    pair_idx: int,
    out_dir: Path,
) -> None:
    """Save a 5-panel figure for one frame pair."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    axes[0].imshow(prev_frame, cmap="gray")
    axes[0].set_title("Frame N (prev)", fontsize=9)

    axes[1].imshow(curr_frame, cmap="gray")
    axes[1].set_title("Frame N+1 (curr)", fontsize=9)

    im2 = axes[2].imshow(of_map, cmap=CMAP_OF, vmin=0, vmax=1)
    axes[2].set_title("Optical Flow\n(motion magnitude)", fontsize=9)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(tm_map, cmap=CMAP_TM, vmin=0, vmax=1)
    axes[3].set_title("Template Matching\n(displacement)", fontsize=9)
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    diff = of_map - tm_map
    im4 = axes[4].imshow(diff, cmap=CMAP_DIFF, vmin=-1, vmax=1)
    axes[4].set_title("Difference\n(OF − TM)", fontsize=9)
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Frame pair {pair_idx:03d}", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / f"pair_{pair_idx:03d}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_summary_heatmaps(
    results_df: pd.DataFrame,
    mean_of_map: np.ndarray,
    mean_tm_map: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Save:
      • Average motion heat-maps for each algorithm
      • Metric comparison bar chart
      • Timing comparison
      • Per-pair correlation line chart
    """
    # ── 1. Average motion heat-maps ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(mean_of_map, cmap=CMAP_OF, vmin=0, vmax=1)
    axes[0].set_title("Average Optical Flow\nMotion Magnitude", fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mean_tm_map, cmap=CMAP_TM, vmin=0, vmax=1)
    axes[1].set_title("Average Template Matching\nMotion Magnitude", fontsize=10)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    diff = mean_of_map - mean_tm_map
    im2 = axes[2].imshow(diff, cmap=CMAP_DIFF, vmin=-1, vmax=1)
    axes[2].set_title("Difference (OF − TM)\nAverage across all pairs", fontsize=10)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / "summary_average_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Metric comparison bar chart ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = ["mse", "mae", "correlation"]
    titles  = ["MSE (lower → more similar)",
               "MAE (lower → more similar)",
               "Pearson correlation\n(higher → more similar)"]
    colors  = ["#e07b54", "#5b9bd5", "#70ad47"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        val = results_df[metric].mean()
        ax.bar([metric.upper()], [val], color=color, edgecolor="black", width=0.4)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(1.05, val * 1.15))
        ax.set_ylabel("Mean value")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Agreement Metrics – Optical Flow vs Template Matching\n"
                 "(computed per frame pair, averaged over all pairs)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_metrics.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 3. Timing comparison ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(results_df))
    ax.plot(x, results_df["of_time_ms"], label="Optical Flow",      color="#e07b54", linewidth=1.5)
    ax.plot(x, results_df["tm_time_ms"], label="Template Matching",  color="#5b9bd5", linewidth=1.5)
    ax.fill_between(x, results_df["of_time_ms"], alpha=0.15, color="#e07b54")
    ax.fill_between(x, results_df["tm_time_ms"], alpha=0.15, color="#5b9bd5")
    ax.set_xlabel("Frame pair index")
    ax.set_ylabel("Processing time (ms)")
    ax.set_title("Processing Time per Frame Pair")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_timing.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 4. Per-pair correlation line chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, results_df["correlation"], color="#70ad47", linewidth=2, marker="o",
            markersize=4, label="Pearson corr (OF vs TM)")
    ax.axhline(results_df["correlation"].mean(), linestyle="--", color="gray",
               label=f"Mean = {results_df['correlation'].mean():.3f}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Frame pair index")
    ax.set_ylabel("Correlation")
    ax.set_title("Per-Frame-Pair Correlation between Optical Flow and Template Matching")
    ax.legend()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_correlation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── 5. Motion coverage comparison (heatmap style) ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    coverage_data = results_df[["of_motion_coverage", "tm_motion_coverage"]].rename(
        columns={"of_motion_coverage": "Optical Flow", "tm_motion_coverage": "Template Matching"}
    )

    # reshape to 2-D for seaborn heatmap: pairs × algorithm
    heat_vals = coverage_data.values.T   # shape (2, N)
    sns.heatmap(
        heat_vals,
        ax=axes[0],
        cmap="YlOrRd",
        yticklabels=["Optical Flow", "Template Matching"],
        xticklabels=[str(i) for i in range(len(results_df))],
        cbar_kws={"label": "Motion coverage"},
        vmin=0, vmax=1,
    )
    axes[0].set_title("Motion Coverage per Frame Pair\n(fraction of pixels > 10 % threshold)",
                       fontsize=9)
    axes[0].set_xlabel("Frame pair index")
    axes[0].tick_params(axis="x", labelsize=6, rotation=90)

    # bar chart of mean coverage
    means = coverage_data.mean()
    bars = axes[1].bar(means.index, means.values,
                       color=["#e07b54", "#5b9bd5"], edgecolor="black", width=0.4)
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Mean motion coverage")
    axes[1].set_title("Average Motion Coverage\nper Algorithm")
    for spine in ["top", "right"]:
        axes[1].spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / "summary_motion_coverage.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_performance_table(results_df: pd.DataFrame, out_dir: Path) -> None:
    """Print and save a formatted performance summary table."""
    summary = pd.DataFrame({
        "Metric": [
            "Avg processing time (ms)",
            "Avg motion coverage",
            "Avg motion intensity (mean map value)",
        ],
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

    print("\n" + "═" * 58)
    print("  PERFORMANCE SUMMARY")
    print("═" * 58)
    print(summary.to_string(index=False))
    print("\n" + "─" * 58)
    print("  AGREEMENT BETWEEN ALGORITHMS")
    print("─" * 58)
    print(agreement.to_string(index=False))
    print("═" * 58 + "\n")

    summary.to_csv(out_dir / "performance_summary.csv", index=False)
    agreement.to_csv(out_dir / "agreement_metrics.csv", index=False)
    results_df.to_csv(out_dir / "per_pair_results.csv", index=False)
    print(f"Tables saved to {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loading
# ══════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_frames(n_frames: int = 32,
                               height: int = 240,
                               width: int = 320) -> list[np.ndarray]:
    """
    Generate synthetic grayscale frames that simulate slow-motion video with
    multiple moving objects (circles + rectangles on a textured background).
    Used as a fallback when the HuggingFace dataset is not accessible.
    """
    rng = np.random.default_rng(42)
    frames = []

    # Background: perlin-like noise texture
    base_bg = (rng.integers(30, 80, (height, width), dtype=np.uint8))
    # Low-frequency smooth background
    bg = cv2.GaussianBlur(base_bg, (31, 31), 0)

    # Define a few moving objects
    objects = [
        # (center_x0, center_y0, dx_per_frame, dy_per_frame, radius, brightness, shape)
        (60,  120,  4,  1, 18, 210, "circle"),
        (200, 80,  -3,  2, 14, 180, "circle"),
        (280, 160,  2, -2, 22, 230, "circle"),
        (100, 200, -2, -1, 15, 195, "circle"),
        (160, 40,   3,  3, 12, 170, "rect"),
        (240, 200, -4,  1, 20, 220, "rect"),
    ]

    for t in range(n_frames):
        frame = bg.copy().astype(np.float32)
        # Add subtle per-frame noise
        noise = rng.normal(0, 3, (height, width)).astype(np.float32)
        frame = np.clip(frame + noise, 0, 255)

        for cx0, cy0, dx, dy, r, bright, shape in objects:
            cx = int((cx0 + dx * t) % width)
            cy = int((cy0 + dy * t) % height)
            if shape == "circle":
                cv2.circle(frame, (cx, cy), r,
                           float(bright), -1)
                cv2.circle(frame, (cx, cy), r,
                           float(bright - 30), 2)
            else:
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(width - 1, cx + r), min(height - 1, cy + r)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              float(bright), -1)

        # Add motion blur to simulate slow-motion camera
        kernel = np.zeros((5, 5))
        kernel[2, :] = 1.0 / 5
        frame = cv2.filter2D(frame, -1, kernel)

        frames.append(np.clip(frame, 0, 255).astype(np.uint8))

    return frames


def _load_from_huggingface(max_pairs: int | None) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Attempt to load frames from the HuggingFace dataset. Returns None on failure."""
    try:
        from datasets import load_dataset as _load_dataset
        print(f"Loading dataset: {DATASET_NAME} …")
        ds = _load_dataset(DATASET_NAME, split="train")
        print(f"  → {len(ds)} samples found.")

        img_cols = [k for k, v in ds.features.items()
                    if "Image" in str(type(v))]
        if not img_cols:
            sample = ds[0]
            img_cols = [k for k, v in sample.items() if isinstance(v, Image.Image)]

        if not img_cols:
            print("  [WARN] No image column detected; switching to synthetic data.")
            return None

        img_col = img_cols[0]
        print(f"  → Using image column: '{img_col}'")

        frames = []
        limit = (max_pairs + 1) if max_pairs else len(ds)
        for i in range(min(limit, len(ds))):
            raw = ds[i][img_col]
            if isinstance(raw, Image.Image):
                gray = pil_to_gray(raw)
            elif isinstance(raw, np.ndarray):
                gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
            else:
                continue
            frames.append(resize_gray(gray))

        if len(frames) < 2:
            return None

        pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        print(f"  → {len(pairs)} consecutive frame-pair(s) prepared.")
        return pairs

    except Exception as exc:
        print(f"  [WARN] HuggingFace dataset unavailable ({exc}); using synthetic data.")
        return None


def load_frame_pairs(max_pairs: int | None = MAX_PAIRS):
    """
    Load the HuggingFace dataset and return a list of (prev_gray, curr_gray)
    numpy pairs.  Falls back to synthetic frame generation when the dataset
    cannot be downloaded (e.g. offline / sandboxed environment).
    All frames are resized to ≤ 320 px in the longest dimension.
    """
    pairs = _load_from_huggingface(max_pairs)

    if pairs is None:
        n_frames = (max_pairs + 1) if max_pairs else 32
        print(f"Generating {n_frames} synthetic frames …")
        frames = _generate_synthetic_frames(n_frames)
        pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
        print(f"  → {len(pairs)} synthetic frame-pair(s) prepared.")

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs_dir = RESULTS_DIR / "pairs"
    pairs_dir.mkdir(exist_ok=True)

    pairs = load_frame_pairs(MAX_PAIRS)

    records      = []
    of_maps_all  = []
    tm_maps_all  = []

    print(f"\nProcessing {len(pairs)} frame pair(s) …\n")
    for idx, (prev, curr) in enumerate(tqdm(pairs, unit="pair")):

        # ── Optical Flow ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        of_map = optical_flow_motion_map(prev, curr)
        of_time_ms = (time.perf_counter() - t0) * 1_000

        # ── Template Matching ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        tm_map = template_matching_motion_map(prev, curr)
        tm_time_ms = (time.perf_counter() - t0) * 1_000

        # resize tm_map to of_map shape (same frame, should already match)
        if tm_map.shape != of_map.shape:
            tm_map = cv2.resize(tm_map, (of_map.shape[1], of_map.shape[0]))

        # ── Metrics ───────────────────────────────────────────────────────────
        m = compute_metrics(of_map, tm_map)
        records.append(dict(
            pair_idx=idx,
            of_time_ms=of_time_ms,
            tm_time_ms=tm_time_ms,
            of_mean_intensity=float(of_map.mean()),
            tm_mean_intensity=float(tm_map.mean()),
            **m,
        ))

        of_maps_all.append(of_map)
        tm_maps_all.append(tm_map)

        # ── Per-pair heat-map figure ──────────────────────────────────────────
        save_pair_heatmap(prev, curr, of_map, tm_map, idx, pairs_dir)

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
