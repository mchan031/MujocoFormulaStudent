# plot_tracks.py
# Run this from your MujocoFormulaStudent root directory
# Output: training_tracks.png  (ready for your thesis figure)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

TRACKS_DIR = "mujoco_tracks"
TRACK_FOLDERS = ["track_1", "track_2", "track_3", "track_4", "track_5"]


def load_cones(track_folder):
    """Load cone positions from random_track.csv."""
    csv_path = os.path.join(TRACKS_DIR, track_folder, "random_track.csv")
    df = pd.read_csv(csv_path, header=None,
                     names=["color", "x", "y", "z", "sx", "sy", "sz"])
    blue   = df[df["color"] == "blue"][["x", "y"]].values
    yellow = df[df["color"] == "yellow"][["x", "y"]].values
    orange = df[df["color"] == "big_orange"][["x", "y"]].values
    return blue, yellow, orange


def load_centreline(track_folder):
    """Load centreline and compute lap length."""
    csv_path = os.path.join(TRACKS_DIR, track_folder, "centreline.csv")
    df = pd.read_csv(csv_path, header=None, skiprows=1)
    # try first two columns as x, y
    pts = df.iloc[:, :2].values.astype(float)
    # compute arc length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    lap_length = float(np.sum(seg_lengths))
    return pts, lap_length


def plot_track(ax, track_folder, title):
    """Plot one track on a given axes."""
    blue, yellow, orange = load_cones(track_folder)
    centreline, lap_length = load_centreline(track_folder)

    # ---- track boundaries (closed loop lines) ----
    def closed(pts):
        return (
            np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1])
        )

    if len(blue) > 1:
        bx, by = closed(blue)
        ax.plot(bx, by, color="#2166ac", linewidth=1.2,
                linestyle="-", zorder=2)

    if len(yellow) > 1:
        yx, yy = closed(yellow)
        ax.plot(yx, yy, color="#d4a017", linewidth=1.2,
                linestyle="-", zorder=2)

    # ---- centreline (dashed) ----
    ax.plot(centreline[:, 0], centreline[:, 1],
            color="#888888", linewidth=0.7,
            linestyle="--", zorder=1, alpha=0.6)

    # ---- cone markers ----
    if len(blue) > 0:
        ax.scatter(blue[:, 0], blue[:, 1],
                   c="#2166ac", s=6, zorder=3)
    if len(yellow) > 0:
        ax.scatter(yellow[:, 0], yellow[:, 1],
                   c="#d4a017", s=6, zorder=3)
    if len(orange) > 0:
        ax.scatter(orange[:, 0], orange[:, 1],
                   c="#e66100", s=20, marker="^", zorder=4,
                   label="Start/Finish")

    # ---- formatting ----
    ax.set_title(f"{title}\nLap length: {lap_length:.0f} m",
                 fontsize=9, pad=4)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]", fontsize=7)
    ax.set_ylabel("Y [m]", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Training Tracks — MuJoCo Formula Student Driverless Simulator",
                 fontsize=12, fontweight="bold", y=0.98)

    # 3 top + 2 bottom layout using GridSpec
    gs = GridSpec(2, 6, figure=fig,
                  hspace=0.45, wspace=0.35)

    # top row: 3 tracks spanning 2 columns each
    axes_top = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]

    # bottom row: 2 tracks centred, spanning 3 columns each
    axes_bot = [
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    all_axes = axes_top + axes_bot

    for ax, folder in zip(all_axes, TRACK_FOLDERS):
        # get track number from folder name
        track_num = folder.split("_")[1]
        plot_track(ax, folder, f"Track {track_num}")

    # shared legend at bottom
    legend_elements = [
        mpatches.Patch(color="#2166ac", label="Blue cones (left boundary)"),
        mpatches.Patch(color="#d4a017", label="Yellow cones (right boundary)"),
        plt.Line2D([0], [0], color="#888888", linestyle="--",
                   linewidth=1, label="Centreline"),
        plt.Line2D([0], [0], marker="^", color="w",
                   markerfacecolor="#e66100", markersize=8,
                   label="Start / Finish"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.01),
               frameon=False)

    plt.savefig("training_tracks.png",
                dpi=200, bbox_inches="tight",
                facecolor="white")
    print("Saved: training_tracks.png")
    plt.show()


if __name__ == "__main__":
    main()