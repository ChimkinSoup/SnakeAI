import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to avoid GUI crashes with pygame
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import sys
import subprocess

# Keep a single figure and write to disk; no GUI window needed.
_fig, _ax = plt.subplots()
_OUT_PATH = os.path.join(os.path.dirname(__file__), "training_plot.png")
_OPEN_AFTER_SAVE = False  # disable auto-opening the first saved plot
_opened_once = False


def _open_image_once():
    global _opened_once
    if _opened_once or not _OPEN_AFTER_SAVE:
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(_OUT_PATH)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", _OUT_PATH])
        else:
            subprocess.Popen(["xdg-open", _OUT_PATH])
        _opened_once = True
    except Exception:
        # Best-effort; don't crash training if opening fails
        pass

def plot(scores, meanScores):
    # Safety: nothing to plot yet.
    if not scores:
        return
    _ax.clear()
    _ax.set_title("Training")
    _ax.set_xlabel("Number of games")
    _ax.set_ylabel("Score")

    xs = list(range(1, len(scores) + 1))
    _ax.plot(xs, scores, label="Score")
    _ax.plot(xs, meanScores, label="Mean score")

    # Force integer ticks only (no half games) and keep them natural numbers.
    _ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, min_n_ticks=1))
    _ax.set_xlim(left=1, right=max(2, len(scores)))  # avoid identical limits when len=1

    _ax.legend()
    _ax.grid(True, linestyle="--", alpha=0.4)

    _fig.tight_layout()
    _fig.savefig(_OUT_PATH)
    _open_image_once()
    