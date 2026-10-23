# HierGAT for Human-Object Interaction (HOI) Detection on Construction Sites

This project is based on the [original HierGAT repository](https://github.com/wjx1198/HierGAT).

## Requirements
- CUDA ≥ 11.8 (the primary setup targets Linux; CPU and MPS are also supported on other platforms)

- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python and dependency management. If `pip` is installed, you can install `uv` by running:
    ```bash
    pip install uv
    ```

## Setup
Set up the Python virtual environment and install the project dependencies:
```bash
uv sync
```

## Data Preparation
1. Download the [datasets](#) and create a `data/` directory to store them:
   ```bash
   mkdir -p data
   ```

2. Create `conf/data/dataset_dirs/construction_hoi.yaml` by copying the example configuration file:
    ```bash
    cp conf/data/dataset_dirs/construction_hoi.example.yaml conf/data/dataset_dirs/construction_hoi.yaml
    ```

3. Edit `conf/data/dataset_dirs/construction_hoi.yaml` to replace the placeholder paths with the paths to the downloaded dataset directories under `data/` (one path per dataset).

## Model Training and Evaluation
Start training by running:
```bash
uv run train_construction_hoi.py
```
By default, Hydra loads the configuration from `conf/config_construction_hoi.yaml` (along with `conf/models/2G-GCN_construction_hoi.yaml` and `conf/data/construction_hoi.yaml`). 

To customize the training settings, edit these configuration files or override individual parameters from the command line:
```bash
# Use a different configuration (e.g. one of the variants in conf/)
uv run train_construction_hoi.py --config-name config_construction_hoi_ms0

# Override training settings
uv run train_construction_hoi.py \
    train.k_fold=10 \
    train.selected_folds=[1,3] \
    train.seed=123

# Override model or optimization settings
uv run train_construction_hoi.py \
    models.optimization.epochs=20 \
    models.optimization.batch_size=64
```

## Visualization
Generate per-fold confusion matrices from a training log:
```
uv run visualize_confusion_matrix.py visualize.log_path=<train.log>
```

## Advanced Usage
<details>
<summary>Click to expand</summary>
  
### Fold Evaluation

Evaluation settings are defined in the `eval` section of `conf/config_construction_hoi.yaml` and can be overridden from the command line.
```bash
# Evaluate a checkpoint on a single dataset
uv run evaluate_fold_on_single_dataset.py \
    eval.checkpoint_path=<ckpt.pt> \
    eval.single_dataset_dir=<features_dir>

# Evaluate a checkpoint on multiple datasets (aggregated metrics)
uv run evaluate_fold_on_multi_dataset_aggregated.py \
    eval.checkpoint_path=<ckpt.pt> \
    eval.multi_dataset_dirs=[/path/a/features,/path/b/features] \
    eval.exclude_video_ids=[C0071,C0074]

# Evaluate each dataset separately
uv run evaluate_fold_on_multi_dataset.py \
    eval.checkpoint_path=<ckpt.pt> \
    eval.resume=true
```

### Prediction
Run prediction using a pretrained model:
```bash
# Use the default prediction settings from the `predict` configuration
uv run predict.py predict.pretrained_model_dir=<model_dir>

# Run prediction with cross-validation enabled
uv run predict.py predict.pretrained_model_dir=<model_dir> predict.cross_validate=true
```
Note: `predict.py` loads the model and data settings from the saved Hydra configuration at `<model_dir>/.hydra/config.yaml`, which is preserved from training time. It does not load these settings from the configuration files in `conf/`.

</details>