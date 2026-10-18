import re
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

from constants import VIS_ACTION_CLASSES


def parse_log_file(log_path: str):
    try:
        with open(log_path) as f:
            content = f.read()
    except OSError as e:
        raise RuntimeError(f"Failed to read log file '{log_path}': {e}") from e

    fold_pattern = re.compile(
        r"Fold (\d+):.*?Test video ID: (\w+).*?"
        r"Confusion matrix:\s*\n(\[\[.*?\]\])",
        re.DOTALL,
    )

    matches = fold_pattern.findall(content)

    folds = []
    for fold_num, test_video_id, cm_str in matches:
        cm_array = parse_confusion_matrix(cm_str)
        folds.append(
            {
                "fold_num": int(fold_num),
                "test_video_id": test_video_id,
                "confusion_matrix": cm_array,
            }
        )

    return folds


def parse_confusion_matrix(cm_str: str) -> np.ndarray:
    lines = cm_str.strip().split("\n")
    rows = []
    for line in lines:
        numbers = re.findall(r"\d+", line)
        rows.append([int(n) for n in numbers])
    return np.array(rows)


def plot_confusion_matrix(
    cm: np.ndarray, action_classes: list, output_path: str, fold_num: int
):
    fig, ax = plt.subplots(figsize=(10, 8))

    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=action_classes,
        yticklabels=action_classes,
        cbar_kws={"label": "Percentage", "format": mtick.PercentFormatter(xmax=1)},
        ax=ax,
        linewidths=0,
        vmin=0,
        vmax=1,
    )

    row_sums = cm.sum(axis=1)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            row_sum = int(row_sums[i])

            cell_intensity = cm_normalized[i, j]
            text_color = "white" if cell_intensity > 0.5 else "black"

            ax.text(
                j + 0.5,
                i + 0.45,
                f"{count} / {row_sum}" if row_sum > 0 else f"{count}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

            if row_sum > 0:
                percentage = (count / row_sum) * 100
                if percentage == int(percentage):
                    percentage_str = f"({int(percentage)}%)"
                else:
                    percentage_str = f"({percentage:.1f}%)"
                ax.text(
                    j + 0.5,
                    i + 0.55,
                    percentage_str,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=text_color,
                )

    ax.set_xlabel("Prediction", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(f"Confusion Matrix - Fold {fold_num}", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@hydra.main(config_path="conf", config_name="config_construction_hoi", version_base=None)
def main(cfg: DictConfig):
    """Plot per-fold confusion matrices from a training log file.

    The configuration is loaded by Hydra from conf/config_construction_hoi.yaml.
    Usage examples:

        python visualize_confusion_matrix.py visualize.log_path=<train.log>
        python visualize_confusion_matrix.py visualize.log_path=<train.log> \
            visualize.output_dir=<output_dir>
    """

    log_path = cfg.visualize.get("log_path", None)
    if not log_path:
        raise ValueError(
            "No log file specified. Set visualize.log_path=<path>."
        )
    output_dir = cfg.visualize.get("output_dir", None)

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    log_name = log_path.stem

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("visualizations") / "confusion_matrices" / log_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing log file: {log_path}")
    folds = parse_log_file(str(log_path))
    print(f"Found {len(folds)} folds")

    for fold in folds:
        fold_num = fold["fold_num"]
        test_video_id = fold["test_video_id"]
        cm = fold["confusion_matrix"]

        filename = f"fold{fold_num:02d}_{test_video_id}_confusion_matrix.png"
        output_path = output_dir / filename

        plot_confusion_matrix(cm, VIS_ACTION_CLASSES, str(output_path), fold_num)
        print(f"Saved: {output_path}")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
