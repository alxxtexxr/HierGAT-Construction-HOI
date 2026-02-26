# AGENTS.md - HierGAT Construction-HOI Codebase Guide

## Overview

This is a PyTorch-based Human-Object Interaction (HOI) detection project for construction sites, based on HierGAT. The project uses Hydra for configuration management and TensorBoard for logging.

## 1. Build/Lint/Test Commands

### Installation
```bash
pip install -r requirements.txt
```

### Running Training
```bash
# Default config
python train.py

# Specific config (via Hydra)
python train_construction_hoi.py conf/config_construction_hoi.yaml
python train_mphoi_14.py

# With custom YAML config
python train.py conf/config_mphoi_14.yaml
```

### Running Tests
**No formal test suite exists.** To add tests, use pytest:
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_file.py

# Run single test function
pytest tests/test_file.py::test_function_name

# Run with verbose output
pytest -v tests/
```

### Code Quality Tools
The project does not have formal linting configured. If adding linting:
```bash
# ruff (recommended for Python)
ruff check .

# flake8
flake8 .

# pylint
pylint src/ pyrutils/

# mypy type checking
mypy src/ pyrutils/
```

### Shell Scripts
```bash
# Run multiple training jobs (example from tt_mphoi_2025.sh)
bash tt_mphoi_2025.sh
```

---

## 2. Code Style Guidelines

### Imports
Organize imports in three sections (recommended):
1. Standard library (`os`, `re`, `datetime`, `pathlib`, `typing`)
2. Third-party packages (`torch`, `numpy`, `hydra`, `omegaconf`)
3. Local modules (`pyrutils`, `vhoi`)

```python
# Example import order
import os
import re
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
import hydra

from pyrutils.torch.train_utils import train, save_checkpoint
from pyrutils.torch.multi_task import MultiTaskLossLearner
from vhoi.data_loading import load_training_data
```

### Formatting
- Line length: 100-120 characters recommended
- Indentation: 4 spaces (no tabs)
- Use Black or Ruff for auto-formatting:
  ```bash
  black .
  ruff format .
  ```

### Naming Conventions
- **Functions/variables**: `snake_case` (e.g., `train_model`, `learning_rate`, `feature_dirs`)
- **Classes**: `PascalCase` (e.g., `GeoGCN`, `MultiTaskLossLearner`, `DataLoader`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `ACTION_CLASSES`, `FEATURE_DIRS`)
- **Private methods/variables**: `_leading_underscore` (e.g., `_internal_method`)

```python
# Good
def train_model(model, optimizer, criterion, epochs):
    learning_rate = 0.001
    return model

class HierGATModel(nn.Module):
    def __init__(self):
        self._hidden_size = 128
        
# Avoid
def TrainModel():  # Not PascalCase for functions
    variable = 1   # Not descriptive
```

### Type Hints
Use type hints for function signatures and variable declarations:

```python
# Recommended
def get_feature_dirs_df(
    feature_dirs, 
    action_classes, 
    new_action_classes: Optional[List[str]] = None
) -> pd.DataFrame:
    ...
    
def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: callable,
    epochs: int,
    device: str,
    loss_names: List[str],
    clip_gradient_at: float = 0.0,
) -> Dict:
    ...
```

### Docstrings
Use Google-style or NumPy-style docstrings for functions and classes:

```python
def train(model, train_loader, optimizer, criterion, epochs, device, loss_names):
    """General training function to train a PyTorch model.

    If validation data is not given, the returned checkpoint is the one obtained 
    after training the model for the specified number of epochs.

    Args:
        model: PyTorch model.
        train_loader: Batch generator for model training.
        optimizer: Model optimizer.
        criterion: Specific loss function.
        epochs: Maximum number of epochs for model training.
        device: Which device to use (cuda or cpu).
        loss_names: Names for the individual losses.

    Returns:
        A dictionary containing training history, model weights, and epoch info.
    """
```

### Error Handling
Use explicit exception handling with meaningful messages:

```python
# Good
try:
    action_label = new_action_classes.index(action_class)
except ValueError:
    action_label = -1

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Set resources.use_gpu: false in config.")

# Avoid bare except
try:
    # risky code
except:
    pass  # Never do this
```

### PyTorch Patterns
Follow these PyTorch conventions:
- Use `nn.Module` base class for models
- Define `forward()` method for all models
- Use `model.to(device)` for device placement
- Use `torch.set_num_threads()` for CPU optimization
- Use `model.parameters()` for optimizer setup
- Use `model.state_dict()` and `model.load_state_dict()` for checkpoints

```python
class GeoGCN(nn.Module):
    def __init__(self, node_n, in_channels, out_channels):
        super(GeoGCN, self).__init__()
        self.weight = Parameter(torch.FloatTensor(64, out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, x):
        # Implementation
        return x
```

### Configuration (Hydra/YAML)
- Store configs in `conf/` directory
- Use `.yaml` files with Hydra defaults
- Follow existing config patterns:

```yaml
defaults:
  - models: 2G-GCN_stage1_mphoi
  - data: mphoi_14

hydra:
  run:
    dir: outputs_hiergat/${data.name}/${metadata.model_name}/${logging.checkpoint_name}

resources:
  use_gpu: true
  num_threads: 32
```

### File Organization
- `src/`: Core utilities and constants
- `pyrutils/`: Reusable utility functions (torch, metrics, geometric)
- `conf/`: Configuration files (models, data, config)
- `train_*.py`: Dataset-specific training scripts

### Logging
- Use `tensorboard` for training metrics (SummaryWriter)
- Print epoch progress to console
- Log to `logs/` directory

---

## Key Dependencies

- torch (2.7.1+cu118)
- hydra-core (1.3.2)
- omegaconf (2.3.0)
- tensorboard (2.20.0)
- numpy, scipy, scikit-learn
- matplotlib, pandas
- zarr, fire

---

## Common Tasks

### Add a new model
1. Create model class in `pyrutils/torch/models_*.py`
2. Add config in `conf/models/`
3. Register in `select_model()` function

### Add new dataset
1. Add config in `conf/data/`
2. Implement data loading in `vhoi.data_loading`
3. Register in data selection functions

### Run single experiment
```bash
python train.py data.name=mphoi14 model.name=2G-GCN
```
