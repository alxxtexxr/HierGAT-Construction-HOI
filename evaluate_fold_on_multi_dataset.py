import os
import re
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig

from utils import get_dataset_dirs


@hydra.main(config_path="conf", config_name="config_construction_hoi", version_base=None)
def main(cfg: DictConfig):
    """Run evaluate_fold_on_single_dataset.py on every dataset directory.

    The configuration is loaded by Hydra from conf/config_construction_hoi.yaml.
    Usage examples:

        python evaluate_fold_on_multi_dataset.py eval.checkpoint_path=<ckpt.pt>
        python evaluate_fold_on_multi_dataset.py \
            eval.checkpoint_path=<ckpt.pt> eval.resume=true
    """

    checkpoint = cfg.eval.get("checkpoint_path", None)
    if not checkpoint:
        raise ValueError(
            "No checkpoint specified. Set eval.checkpoint_path=<path>."
        )
    resume = cfg.eval.get("resume", False)  # Re-evaluate datasets already done

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    fold_match = re.search(r"fold(\d+)", checkpoint_path.name)
    if fold_match:
        fold_number = int(fold_match.group(1))
    else:
        fold_number = 0

    checkpoint_name = checkpoint_path.stem
    checkpoint_base = re.sub(r"_fold\d+$", "", checkpoint_name)

    datasets = [str(p) for p in get_dataset_dirs(cfg.data.data_dir)]

    total = len(datasets)
    skipped = 0
    evaluated = 0
    errors = 0

    for i, dataset in enumerate(datasets, 1):
        dataset_path = Path(dataset)
        video_id_match = re.search(r"([A-Z]\d+)", dataset_path.parent.stem)
        video_id = video_id_match.group(1) if video_id_match else "unknown"

        eval_dir = f"{os.getcwd()}/eval/{checkpoint_base}/fold{fold_number:02d}/{video_id}"

        if os.path.exists(eval_dir) and not resume:
            print(f"[{i}/{total}] Skipping {video_id} (already evaluated)")
            skipped += 1
            continue

        print(f"\n{'='*64}")
        print(f"[{i}/{total}] Evaluating dataset: {video_id}")
        print(f"{'='*64}\n")

        cmd = [
            "conda", "run", "-n", "py39_torch271", "--no-capture-output",
            "python", "evaluate_fold_on_single_dataset.py",
            f"eval.checkpoint_path={checkpoint}",
            f"eval.single_dataset_dir={dataset}",
        ]

        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"Error evaluating {dataset}")
            errors += 1
        else:
            print(f"Completed: {video_id}")
            evaluated += 1

    print(f"\n{'='*64}")
    print("Summary:")
    print(f"  Total datasets: {total}")
    print(f"  Evaluated: {evaluated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
