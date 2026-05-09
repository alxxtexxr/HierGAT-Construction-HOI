import os
import re
import json
import gc
import random
from pathlib import Path
from datetime import datetime

import fire
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from pyrutils.torch.train_utils import save_checkpoint
from vhoi.data_loading import (
    input_size_from_data_loader,
    select_model_data_feeder,
    select_model_data_fetcher,
)
from vhoi.data_loading_custom import create_data, create_data_loader
from vhoi.losses_custom_v2 import (
    select_loss,
    decide_num_main_losses,
    select_loss_types,
    select_loss_learning_mask,
)
from vhoi.models import load_model_weights
from vhoi.models_custom_v2 import TGGCN

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from predict import match_shape, match_att_shape
from src.constants import (
    ACTION_CLASSES_V2,
    NEW_ACTION_CLASSES_V2,
    VIS_ACTION_CLASSES_V2,
    FEATURE_DIRS,
)
from src.utils import printh, get_feature_dirs_df, create_data_df

torch.multiprocessing.set_sharing_strategy("file_system")


def main(
    checkpoint: str,
    dataset: str,
    config_path: str = "conf/config_construction_hoi.yaml",
):
    checkpoint_path = Path(checkpoint)
    dataset_path = Path(dataset)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    fold_match = re.search(r"fold(\d+)", checkpoint_path.name)
    if fold_match:
        fold_number = int(fold_match.group(1))
    else:
        fold_number = 0

    timestamp_match = re.search(r"(\d{12})", checkpoint_path.name)
    timestamp = timestamp_match.group(1) if timestamp_match else "unknown"

    video_id_match = re.search(r"([A-Z]\d+)", dataset_path.parent.stem)
    video_id = video_id_match.group(1) if video_id_match else "unknown"

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Timestamp: {timestamp}")
    print(f"Fold: {fold_number}")
    print(f"Video ID: {video_id}")
    print()

    random.seed(42)
    os.environ["PYTHONHASHSEED"] = "42"
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config_path = Path(config_path)
    with hydra.initialize(config_path=str(config_path.parent), version_base=None):
        cfg: DictConfig = hydra.compose(config_name=str(config_path.name))

    torch.set_num_threads(cfg.resources.num_threads)
    model_name, model_input_type = (
        cfg.models.metadata.model_name,
        cfg.models.metadata.input_type,
    )
    batch_size = cfg.models.optimization.batch_size
    misc_dict = cfg.get("misc", default_value={})
    sigma = misc_dict.get("segmentation_loss", {}).get("sigma", 0.0)
    scaling_strategy = cfg.data.scaling_strategy
    downsampling = cfg.data.downsampling

    checkpoint_name = checkpoint_path.stem
    checkpoint_base = re.sub(r"_fold\d+$", "", checkpoint_name)
    eval_dir = f"{os.getcwd()}/eval/{checkpoint_base}/fold{fold_number:02d}/{video_id}"
    os.makedirs(eval_dir, exist_ok=True)
    print(f"Eval directory: {eval_dir}")
    print()

    printh("Loading Checkpoint", 96)

    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    scalers = checkpoint_data.get("scalers", None)
    start_epoch = checkpoint_data.get("epoch", 0)
    print(f"Checkpoint epoch: {start_epoch}")
    print()

    printh("Loading Data", 96)

    feature_dirs_df = get_feature_dirs_df(
        FEATURE_DIRS,
        ACTION_CLASSES_V2,
        NEW_ACTION_CLASSES_V2,
    )
    feature_dirs_df = feature_dirs_df.dropna(subset=['action_label'])

    test_feature_dirs_df = feature_dirs_df[
        feature_dirs_df["base_dir"] == str(dataset_path)
    ]

    if len(test_feature_dirs_df) == 0:
        test_feature_dirs_df = feature_dirs_df[
            feature_dirs_df["base_dir"].str.contains(dataset_path.stem, na=False)
        ]

    print(f"Test feature dirs count: {len(test_feature_dirs_df)}")
    print()

    test_data = create_data(
        test_feature_dirs_df["dir"].tolist(),
        ACTION_CLASSES_V2,
        NEW_ACTION_CLASSES_V2,
    )

    test_loader, _, _ = create_data_loader(
        *test_data,
        model_name,
        batch_size=len(test_data[0]),
        shuffle=False,
        scalers=scalers,
        sigma=sigma,
        downsampling=downsampling,
    )

    print(f"Test data size: {len(test_data[0])}")
    print()

    printh("Creating Model", 96)

    input_size = input_size_from_data_loader(test_loader, model_name, model_input_type)
    data_info = {"input_size": input_size}

    model_creation_args = cfg.models.parameters
    model_creation_args = {**data_info, **model_creation_args}
    dataset_name = cfg.data.name
    num_classes = len(NEW_ACTION_CLASSES_V2)
    model_creation_args["num_classes"] = (num_classes, None)

    device = "cuda" if torch.cuda.is_available() and cfg.resources.use_gpu else "cpu"
    print(f"Device: {device}")

    model = TGGCN(feat_dim=1024, **model_creation_args).to(device)

    if "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
    else:
        state_dict = checkpoint_data["state_dict"]

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")

    print("Model loaded successfully")
    print()

    fetch_model_data = select_model_data_fetcher(
        model_name,
        model_input_type,
        dataset_name=dataset_name,
        **{**misc_dict, **cfg.models.parameters.__dict__},
    )
    feed_model_data = select_model_data_feeder(
        model_name, model_input_type, dataset_name=dataset_name, **misc_dict
    )
    num_main_losses = decide_num_main_losses(
        model_name, dataset_name, {**misc_dict, **cfg.models.parameters.__dict__}
    )

    printh("Evaluation", 96)

    model.eval()

    outputs, targets = [], []

    for i, data in enumerate(test_loader):
        data, target = fetch_model_data(data, device=device)
        with torch.no_grad():
            output = feed_model_data(model, data)

        if num_main_losses is not None:
            output = output[-num_main_losses:]
            target = target[-num_main_losses:]

        outputs += output
        targets += target

    y_pred = torch.argmax(outputs[0], dim=1).cpu().numpy()
    y_true = targets[0].squeeze(-1).mode(dim=1).values.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print("=" * 64)
    print("Evaluation Results")
    print("=" * 64)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("=" * 64)
    print()

    ticklabels = VIS_ACTION_CLASSES_V2
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ticklabels))))

    print("Confusion Matrix:")
    print(cm)
    print()

    fig, ax = plt.subplots(figsize=(10, 8))

    cm_max = cm.max()
    if cm_max > 0:
        cm_normalized = cm.astype(float) / cm_max
    else:
        cm_normalized = cm.astype(float)

    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cbar_kws={"label": "Count"},
        ax=ax,
        linewidths=0,
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
                ax.text(
                    j + 0.5,
                    i + 0.55,
                    f"({percentage:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=text_color,
                )

    ax.set_xlabel("Prediction", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(f"Confusion Matrix - Fold {fold_number}", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(f"{eval_dir}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {eval_dir}/confusion_matrix.png")
    print()

    metrics = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "timestamp": timestamp,
        "fold": fold_number,
        "video_id": video_id,
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "class_labels": VIS_ACTION_CLASSES_V2,
    }

    with open(f"{eval_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"Metrics saved to {eval_dir}/metrics.json")
    print()

    print("Evaluation complete!")


if __name__ == "__main__":
    fire.Fire(main)