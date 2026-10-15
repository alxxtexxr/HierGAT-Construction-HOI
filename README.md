# HierGAT Model for Human-Object Interaction (HOI) Detection in Construction Sites

https://github.com/user-attachments/assets/e50d8061-f5ad-4bcd-bf3b-2ad013e10e81

Original HierGAT repository: https://github.com/wjx1198/HierGAT

## Setup
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## Model Training and Evaluation
```bash
python train_construction_hoi.py
```
The configuration is loaded by Hydra from `conf/config_construction_hoi.yaml` (along with `conf/models/2G-GCN_construction_hoi.yaml` and `conf/data/construction_hoi.yaml`). To customize the training settings, modify these files or override values from the command line:
```bash
# Use a different config (e.g. one of the variants in conf/)
python train_construction_hoi.py --config-name config_construction_hoi_lr1e-3

# Override settings from any section, e.g. training run settings
python train_construction_hoi.py train.k_fold=5 train.selected_folds=[1,3] train.seed=123

# Or model/optimization settings
python train_construction_hoi.py models.optimization.epochs=20 models.optimization.batch_size=64
```

## Model Evaluation
Evaluation settings live in the `eval` section of the config (see `conf/config_construction_hoi.yaml`) and are overridden from the command line:
```bash
# Evaluate a checkpoint on a single dataset
python evaluate_fold_on_single_dataset.py \
    eval.checkpoint_path=<ckpt.pt> eval.single_dataset_dir=<features_dir>

# Evaluate a checkpoint on multiple datasets (aggregated metrics)
python evaluate_fold_on_multi_dataset_aggregated.py \
    eval.checkpoint_path=<ckpt.pt> \
    eval.multi_dataset_dirs=[/path/a/features,/path/b/features] \
    eval.exclude_video_ids=[C0071,C0074]

# Or run evaluate_fold_on_single_dataset.py once per dataset dir
python evaluate_fold_on_multi_dataset.py eval.checkpoint_path=<ckpt.pt> eval.resume=true
```

## Prediction and Visualisation
```bash
# Run prediction for a pre-trained model (settings in the `predict` section)
python predict.py predict.pretrained_model_dir=<model_dir>
python predict.py predict.pretrained_model_dir=<model_dir> predict.cross_validate=true

# Plot per-fold confusion matrices from a training log (settings in `visualize`)
python visualize_confusion_matrix.py visualize.log_path=<train.log>
```
Note: `predict.py` reads model/data settings from the saved Hydra config inside `<model_dir>/.hydra/config.yaml` (frozen at training time), not from `conf/`.
