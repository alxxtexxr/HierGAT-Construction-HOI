import os
import random
import json
import gc
from datetime import datetime
from pathlib import Path

import fire
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from pyrutils.torch.train_utils import train, save_checkpoint
from pyrutils.torch.multi_task import MultiTaskLossLearner
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

torch.multiprocessing.set_sharing_strategy("file_system")  # Avoid shared memory crashes


def main(
    config_path="conf/config_construction_hoi.yaml",
    seed=42,
    k=10,
    selected_folds=None,           # List of fold indices to run (0-based). None = run all k folds.
                                    # Example: selected_folds=[0, 2, 5] to run only folds 0, 2, and 5.
    num_epochs=None,                # DEFAULT: Number of epochs in the configuration file (10).
                                    # With early stopping enabled, this is the maximum number of epochs
    batch_size=None,                # DEFAULT: Batch size in the configuration file (32)
    min_epochs=10,                  # Minimum epochs before early stopping can activate
    early_stopping_patience=None,   # Early stopping patience (None = disabled)
    num_workers=4,                  # Number of data loading workers
):
    # Set random seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load configuration
    config_path = Path(config_path)
    with hydra.initialize(config_path=str(config_path.parent), version_base=None):
        cfg: DictConfig = hydra.compose(config_name=str(config_path.name))

    if num_epochs is not None:
        cfg.models.optimization.epochs = num_epochs
    
    if batch_size is not None:
        cfg.models.optimization.batch_size = batch_size

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

    # Determine test feature directories for k-fold
    feature_dirs_df = get_feature_dirs_df(
        feature_dirs=FEATURE_DIRS,
        action_classes=ACTION_CLASSES_V2,
        new_action_classes=NEW_ACTION_CLASSES_V2,
    )
    feature_dirs_df = feature_dirs_df.dropna(subset=['action_label'])
    
    feature_dir_counts = (
        feature_dirs_df["base_dir"].value_counts().sort_values()
    )  # Count occurrences of each feature directory
    # and sort from lowest to highest
    # kfold_test_feature_dirs = feature_dir_counts.index[:k] # OLD
    
    # Monte Carlo: Vanilla round-robin sampling of (val, test) pairs
    # Each fold gets 1 val + 1 test dir (different from each other)
    # Dirs can repeat across folds (e.g., dir1 is test in fold0, val in fold1)
    all_dirs = feature_dir_counts.index.tolist()
    if len(all_dirs) < 2:
        raise ValueError(
            f"Need at least 2 feature directories to sample 1 val + 1 test, got {len(all_dirs)}."
        )
    # Generate round-robin pairs deterministically
    num_dirs = len(all_dirs)
    kfold_val_feature_dirs = []
    kfold_test_feature_dirs = []
    for i in range(k):
        val_idx = (i * 2) % num_dirs
        test_idx = (i * 2 + 1) % num_dirs
        kfold_val_feature_dirs.append(all_dirs[val_idx])
        kfold_test_feature_dirs.append(all_dirs[test_idx])

    # Validate and resolve selected_folds (1-based: fold numbers from 1 to k)
    # Handle fire.Fire passing a single int instead of a list
    if isinstance(selected_folds, int):
        selected_folds = [selected_folds]
    if selected_folds is not None:
        if not selected_folds:
            raise ValueError("selected_folds must not be empty.")
        invalid_folds = [f for f in selected_folds if f < 1 or f > k]
        if invalid_folds:
            raise ValueError(
                f"Invalid fold numbers {invalid_folds}. Valid range: 1 to {k}."
            )
    else:
        selected_folds = list(range(1, k + 1))

    # Create evaluation directory
    run_name = (
        f"hs{cfg.models.parameters.hidden_size}_"
        f"e{cfg.models.optimization.epochs}_"
        f"bs{cfg.models.optimization.batch_size}_"
        f"lr{cfg.models.optimization.learning_rate}_"
        f"k{k}_"
        f"{datetime.now().strftime('%y%m%d%H%M%S')}"
    )
    eval_dir = f"{os.getcwd()}/eval/{run_name}"
    try:
        os.makedirs(eval_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create eval directory '{eval_dir}': {e}") from e

    # Iterate through selected folds
    acc_list, f1_list, precision_list, recall_list, cm_list = [], [], [], [], []
    val_acc_list, val_f1_list, val_precision_list, val_recall_list, val_cm_list = [], [], [], [], []
    best_train_loss_list, best_val_loss_list = [], []
    for fold_num in selected_folds:
        fold_idx = fold_num - 1  # Convert 1-based to 0-based index
        test_feature_dir = kfold_test_feature_dirs[fold_idx]
        val_feature_dir = kfold_val_feature_dirs[fold_idx]
        printh(f"Fold {fold_num}: val={val_feature_dir}, test={test_feature_dir}", 128)

        # Create log directory
        root_log_dir = f"{os.getcwd()}/outputs_hiergat/construction_hoi"
        fold_str = str(fold_num).zfill(len(str(k)))
        checkpoint_name = f"{run_name}_fold{fold_str}"
        log_dir = f"{root_log_dir}/{checkpoint_name}"
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory '{log_dir}': {e}") from e

        print("Log directory:", log_dir)
        print()

        printh("Train-Test Splitting", 96)

        # Log training, validation, and test video IDs
        test_video_id = Path(test_feature_dir).parent.stem.split("_")[0]
        val_video_id = Path(val_feature_dir).parent.stem.split("_")[0]
        all_video_ids = [
            feature_dir.parent.stem.split("_")[0] for feature_dir in FEATURE_DIRS
        ]
        train_video_ids = [
            video_id for video_id in all_video_ids
            if video_id != test_video_id and video_id != val_video_id
        ]

        print("Test video ID (unseen):", test_video_id)
        print("Validation video ID (unseen):", val_video_id)
        print("Training video IDs (seen):")
        print(train_video_ids)
        print()
        print("All video IDs (unseen + seen):")
        print(all_video_ids)
        print()

        # Split train, val, and test feature directories
        test_feature_dirs_df = feature_dirs_df[
            feature_dirs_df["base_dir"] == test_feature_dir
        ]
        val_feature_dirs_df = feature_dirs_df[
            feature_dirs_df["base_dir"] == val_feature_dir
        ]
        train_feature_dirs_df = feature_dirs_df[
            ~feature_dirs_df["base_dir"].isin([test_feature_dir, val_feature_dir])
        ]

        print("Training action label counts:")
        print(train_feature_dirs_df["action_label"].value_counts().sort_index())
        print()
        print("Validation action label counts:")
        print(val_feature_dirs_df["action_label"].value_counts().sort_index())
        print()
        print("Test action label counts:")
        print(test_feature_dirs_df["action_label"].value_counts().sort_index())
        print()

        # Create train data dataframe
        df = create_data_df(
            train_feature_dirs_df["dir"].tolist(),
            ACTION_CLASSES_V2,
            NEW_ACTION_CLASSES_V2 if "NEW_ACTION_CLASSES_V2" in globals() else None,
        )

        printh("Train Data", 96)
        print("Training action label info:")
        print(df["action_label"].value_counts().sort_index())
        print("Total:", len(df))
        print()

        # Create train, validation, and test dataframes
        train_df = df  # All training data (no train/val split within training dirs)

        train_size = len(train_df)
        val_size = len(val_feature_dirs_df)
        test_size = len(test_feature_dirs_df)
        all_size = train_size + val_size + test_size

        print(f"Total train data: {train_size} ({train_size / all_size * 100:.2f}%)")
        print(f"Total val data: {val_size} ({val_size / all_size * 100:.2f}%)")
        print(f"Total test data: {test_size} ({test_size / all_size * 100:.2f}%)")
        print()

        data_cols = [
            "human_features",
            "human_boxes",
            "human_poses",
            "object_features",
            "object_boxes",
            "gt",
            "xs_step",
        ]
        train_data = train_df[data_cols].to_numpy().T.tolist()
        val_data = create_data(
            val_feature_dirs_df["dir"].tolist(),
            ACTION_CLASSES_V2,
            NEW_ACTION_CLASSES_V2 if "NEW_ACTION_CLASSES_V2" in globals() else None,
        )
        test_data = create_data(
            test_feature_dirs_df["dir"].tolist(),
            ACTION_CLASSES_V2,
            NEW_ACTION_CLASSES_V2 if "NEW_ACTION_CLASSES_V2" in globals() else None,
        )

        # Create data loaders
        train_loader, scalers, _ = create_data_loader(
            *train_data,
            model_name,
            batch_size=batch_size,
            shuffle=True,
            scaling_strategy=scaling_strategy,
            sigma=sigma,
            downsampling=downsampling,
            num_workers=num_workers,
        )
        val_loader, _, _ = create_data_loader(
            *val_data,
            model_name,
            batch_size=len(val_data[0]),
            shuffle=False,
            scalers=scalers,
            sigma=sigma,
            downsampling=downsampling,
            num_workers=num_workers,
        )
        test_loader, _, _ = create_data_loader(
            *test_data,
            model_name,
            batch_size=len(test_data[0]),
            shuffle=False,
            scalers=scalers,
            sigma=sigma,
            downsampling=downsampling,
            num_workers=num_workers,
        )
        input_size = input_size_from_data_loader(
            train_loader, model_name, model_input_type
        )
        data_info = {"input_size": input_size}

        # Create model
        model_creation_args = cfg.models.parameters
        model_creation_args = {**data_info, **model_creation_args}
        dataset_name = cfg.data.name
        num_classes = len(NEW_ACTION_CLASSES_V2)
        model_creation_args["num_classes"] = (num_classes, None)
        device = (
            "cuda" if torch.cuda.is_available() and cfg.resources.use_gpu else "cpu"
        )
        model = TGGCN(feat_dim=1024, **model_creation_args).to(device)
        if (
            misc_dict.get("pretrained", False)
            and misc_dict.get("pretrained_path") is not None
        ):
            state_dict = load_model_weights(misc_dict["pretrained_path"])
            model.load_state_dict(state_dict, strict=False)
        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=cfg.models.optimization.learning_rate)
        criterion, loss_names = select_loss(
            model_name, model_input_type, dataset_name, cfg=cfg
        )
        mtll_model = None
        if misc_dict.get("multi_task_loss_learner", False):
            loss_types = select_loss_types(model_name, dataset_name, cfg=cfg)
            mask = select_loss_learning_mask(model_name, dataset_name, cfg=cfg)
            mtll_model = MultiTaskLossLearner(loss_types=loss_types, mask=mask).to(
                device
            )
            optimizer.add_param_group({"params": mtll_model.parameters()})
        # Some config + model training
        tensorboard_log_dir = root_log_dir  # OLD: cfg.models.logging.root_log_dir
        # checkpoint_name = cfg.models.logging.checkpoint_name
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

        # Training
        printh("Training", 96)

        checkpoint = train(
            model,
            train_loader,
            optimizer,
            criterion,
            cfg.models.optimization.epochs,
            device,
            loss_names,
            clip_gradient_at=cfg.models.optimization.clip_gradient_at,
            fetch_model_data=fetch_model_data,
            feed_model_data=feed_model_data,
            val_loader=val_loader,
            mtll_model=mtll_model,
            num_main_losses=num_main_losses,
            tensorboard_log_dir=tensorboard_log_dir,
            checkpoint_name=checkpoint_name,
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
        )

        print()

        # Logging
        # if cfg.models.logging.log_dir is not None: # OLD
        if log_dir is not None:
            # log_dir = cfg.models.logging.log_dir
            checkpoint["scalers"] = scalers
            save_checkpoint(
                log_dir,
                checkpoint,
                checkpoint_name=checkpoint_name,
                include_timestamp=False,
            )

        # Load the BEST model weights from checkpoint (based on validation loss)
        # The train() function saves the best model in checkpoint['model_state_dict']
        best_epoch = checkpoint.get('epoch', 0)
        print(f"\nLoading best model from epoch {best_epoch}...")
        model.load_state_dict(checkpoint['model_state_dict'])

        # Extract best epoch losses from checkpoint
        best_epoch_idx = best_epoch - 1  # 0-indexed
        if checkpoint['train_losses'] and best_epoch_idx < len(checkpoint['train_losses']):
            best_train_loss = checkpoint['train_losses'][best_epoch_idx][0]
        else:
            best_train_loss = checkpoint['train_losses'][-1][0]
        if checkpoint['val_losses'] and best_epoch_idx < len(checkpoint['val_losses']):
            best_val_loss = checkpoint['val_losses'][best_epoch_idx][0]
        else:
            best_val_loss = checkpoint['val_losses'][-1][0]
        best_train_loss_list.append(best_train_loss)
        best_val_loss_list.append(best_val_loss)
        print(f"Best epoch train loss: {best_train_loss:.4f}")
        print(f"Best epoch val loss:   {best_val_loss:.4f}")

        # Validation Evaluation
        printh("Validation Evaluation", 96)
        print("Validation video ID:", val_video_id)
        print()

        model.eval()

        inspect_model = False
        val_outputs, val_targets, val_attentions = [], [], []

        for i, dataset in enumerate(val_loader):
            data, target = fetch_model_data(dataset, device=device)
            with torch.no_grad():
                output = feed_model_data(model, data)
            if inspect_model:
                output, attention_scores = output
                attention_scores = [att_score[:, 0] for att_score in attention_scores]
            if num_main_losses is not None:
                output = output[-num_main_losses:]
                target = target[-num_main_losses:]
            if downsampling > 1:
                for i, (out, tgt) in enumerate(zip(output, target, strict=True)):
                    if out.ndim != 4:
                        raise RuntimeError(
                            f"Number of dimensions for output is {out.ndim}"
                        )
                    out = torch.repeat_interleave(out, repeats=downsampling, dim=-2)
                    out = match_shape(out, tgt)
                    output[i] = out
                if inspect_model:
                    a_target = target[0]
                    attention_scores = [
                        torch.repeat_interleave(att_score, repeats=downsampling, dim=-2)
                        for att_score in attention_scores
                    ]
                    attention_scores = [
                        match_att_shape(att_score, a_target)
                        for att_score in attention_scores
                    ]
                    val_attentions.append(attention_scores)
            val_outputs += output
            val_targets += target

        val_y_pred = torch.argmax(val_outputs[0], dim=1).cpu().numpy()
        val_y_true = val_targets[0].squeeze(-1).mode(dim=1).values.cpu().numpy()

        val_acc = accuracy_score(val_y_true, val_y_pred)
        val_f1 = f1_score(val_y_true, val_y_pred, average="weighted", zero_division=0)
        val_precision = precision_score(val_y_true, val_y_pred, average="weighted", zero_division=0)
        val_recall = recall_score(val_y_true, val_y_pred, average="weighted", zero_division=0)

        print("Validation Accuracy:", val_acc)
        print("Validation F1 score:", val_f1)
        print("Validation Precision:", val_precision)
        print("Validation Recall:", val_recall)

        # Plot confusion matrix
        ticklabels = NEW_ACTION_CLASSES_V2

        val_cm = confusion_matrix(val_y_true, val_y_pred, labels=list(range(len(ticklabels))))

        print("Validation Confusion matrix:")
        print(val_cm)
        print()

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            val_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=VIS_ACTION_CLASSES_V2,
            yticklabels=VIS_ACTION_CLASSES_V2,
        )
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.title("Validation Confusion Matrix")

        # Save confusion matrix
        val_cm_dir = f"{eval_dir}/val_cm"
        try:
            os.makedirs(val_cm_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create val_cm directory '{val_cm_dir}': {e}") from e
        plt.savefig(f"{val_cm_dir}/fold{fold_str}.png")
        plt.close()

        # Store validation metrics
        val_acc_list.append(val_acc)
        val_f1_list.append(val_f1)
        val_precision_list.append(val_precision)
        val_recall_list.append(val_recall)
        val_cm_list.append(val_cm)

        del val_outputs, val_targets
        torch.cuda.empty_cache()

        # Test Evaluation
        printh("Test Evaluation", 96)
        print("Test video ID:", test_video_id)
        print()

        outputs, targets, attentions = [], [], []

        for i, dataset in enumerate(test_loader):
            data, target = fetch_model_data(dataset, device=device)
            with torch.no_grad():
                output = feed_model_data(model, data)
            if inspect_model:
                output, attention_scores = output
                attention_scores = [att_score[:, 0] for att_score in attention_scores]
            if num_main_losses is not None:
                output = output[-num_main_losses:]
                target = target[-num_main_losses:]
            if downsampling > 1:
                for i, (out, tgt) in enumerate(zip(output, target, strict=True)):
                    if out.ndim != 4:
                        raise RuntimeError(
                            f"Number of dimensions for output is {out.ndim}"
                        )
                    out = torch.repeat_interleave(out, repeats=downsampling, dim=-2)
                    out = match_shape(out, tgt)
                    output[i] = out
                if inspect_model:
                    a_target = target[0]
                    attention_scores = [
                        torch.repeat_interleave(att_score, repeats=downsampling, dim=-2)
                        for att_score in attention_scores
                    ]
                    attention_scores = [
                        match_att_shape(att_score, a_target)
                        for att_score in attention_scores
                    ]
                    attentions.append(attention_scores)
            outputs += output
            targets += target

        y_pred = torch.argmax(outputs[0], dim=1).cpu().numpy()
        y_true = targets[0].squeeze(-1).mode(dim=1).values.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        print("Test Accuracy:", acc)
        print("Test F1 score:", f1)
        print("Test Precision:", precision)
        print("Test Recall:", recall)

        # Plot confusion matrix
        ticklabels = NEW_ACTION_CLASSES_V2

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ticklabels))))

        print("Test Confusion matrix:")
        print(cm)
        print()

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=VIS_ACTION_CLASSES_V2,
            yticklabels=VIS_ACTION_CLASSES_V2,
        )
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.title("Test Confusion Matrix")

        # Save confusion matrix
        cm_dir = f"{eval_dir}/cm"
        try:
            os.makedirs(cm_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create cm directory '{cm_dir}': {e}") from e
        plt.savefig(f"{cm_dir}/fold{fold_str}.png")
        plt.close()

        # Cleanup (VERY IMPORTANT)
        del model, optimizer
        del train_loader, val_loader, test_loader
        del train_data, val_data, test_data
        del df, train_df
        del outputs, targets
        del val_outputs, val_targets
        torch.cuda.empty_cache()
        gc.collect()

        # Store test metrics
        acc_list.append(acc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        cm_list.append(cm)

    # Compute mean and std for all folds
    acc_mean, acc_std = np.mean(acc_list), np.std(acc_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
    precision_mean, precision_std = np.mean(precision_list), np.std(precision_list)
    recall_mean, recall_std = np.mean(recall_list), np.std(recall_list)

    val_acc_mean, val_acc_std = np.mean(val_acc_list), np.std(val_acc_list)
    val_f1_mean, val_f1_std = np.mean(val_f1_list), np.std(val_f1_list)
    val_precision_mean, val_precision_std = np.mean(val_precision_list), np.std(val_precision_list)
    val_recall_mean, val_recall_std = np.mean(val_recall_list), np.std(val_recall_list)

    best_train_loss_mean, best_train_loss_std = np.mean(best_train_loss_list), np.std(best_train_loss_list)
    best_val_loss_mean, best_val_loss_std = np.mean(best_val_loss_list), np.std(best_val_loss_list)

    print("\n" + "=" * 64)
    print("Training All-Fold Results Summary")
    print("=" * 64)
    print(f"Loss:      {best_train_loss_mean:.4f} ± {best_train_loss_std:.4f}")
    print("=" * 64)

    print("\n" + "=" * 64)
    print("Validation All-Fold Results Summary")
    print("=" * 64)
    print(f"Loss:      {best_val_loss_mean:.4f} ± {best_val_loss_std:.4f}")
    print(f"Accuracy:  {val_acc_mean:.4f} ± {val_acc_std:.4f}")
    print(f"F1 Score:  {val_f1_mean:.4f} ± {val_f1_std:.4f}")
    print(f"Precision: {val_precision_mean:.4f} ± {val_precision_std:.4f}")
    print(f"Recall:    {val_recall_mean:.4f} ± {val_recall_std:.4f}")
    print("=" * 64)

    print("\n" + "=" * 64)
    print("Test All-Fold Results Summary")
    print("=" * 64)
    print(f"Accuracy:  {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"F1 Score:  {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"Recall:    {recall_mean:.4f} ± {recall_std:.4f}")
    print("=" * 64 + "\n")

    # Convert numpy arrays to lists for JSON serialization
    cm_list_serializable = [cm.tolist() for cm in cm_list]
    val_cm_list_serializable = [cm.tolist() for cm in val_cm_list]

    # Save metrics
    metrics = {
        "val": {
            "acc": {
                "folds": val_acc_list,
                "mean": val_acc_mean,
                "std": val_acc_std,
            },
            "f1": {
                "folds": val_f1_list,
                "mean": val_f1_mean,
                "std": val_f1_std,
            },
            "precision": {
                "folds": val_precision_list,
                "mean": val_precision_mean,
                "std": val_precision_std,
            },
            "recall": {
                "folds": val_recall_list,
                "mean": val_recall_mean,
                "std": val_recall_std,
            },
            "cm": {
                "folds": val_cm_list_serializable,
            },
        },
        "test": {
            "acc": {
                "folds": acc_list,
                "mean": acc_mean,
                "std": acc_std,
            },
            "f1": {
                "folds": f1_list,
                "mean": f1_mean,
                "std": f1_std,
            },
            "precision": {
                "folds": precision_list,
                "mean": precision_mean,
                "std": precision_std,
            },
            "recall": {
                "folds": recall_list,
                "mean": recall_mean,
                "std": recall_std,
            },
            "cm": {
                "folds": cm_list_serializable,
            },
        },
    }

    try:
        with open(f"{eval_dir}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
    except OSError as e:
        raise RuntimeError(f"Failed to save metrics to '{eval_dir}/metrics.json': {e}") from e


if __name__ == "__main__":
    fire.Fire(main)
