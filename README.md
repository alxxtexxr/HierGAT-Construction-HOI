# HierGAT Model for Human-Object Interaction (HOI) Detection in Construction Sites

Original HierGAT repository: https://github.com/wjx1198/HierGAT

## Setup
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## Model Training and Evaluation
```bash
python train_construction_hoi.py <CONFIG_PATH>
```
Arguments:
- `<CONFIG_PATH>`: Path to the training configuration file. The default is `conf/config_construction_hoi.yaml`. To customize the training settings, modify this file along with `conf/models/2G-GCN_construction_hoi.yaml` and `conf/data/construction_hoi.yaml`.
