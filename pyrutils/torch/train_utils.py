from datetime import datetime
import os
from typing import Optional
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

from pyrutils.torch.fetchers import single_input_single_output
from pyrutils.torch.forwarders import basic_forward


def _validate_with_metric(model, val_loader, criterion, device, loss_names,
                           fetch_model_data, feed_model_data, metric_name,
                           mtll_model=None, num_main_losses=None, **kwargs):
    """Run a single validation pass that computes both loss and classification metric.

    This avoids running two separate inference loops over the val set.

    Args:
        model: PyTorch model.
        val_loader: Batch generator for model validation.
        criterion: Loss function.
        device: Which device to use.
        loss_names: Names for the individual losses.
        fetch_model_data: Function to fetch input/output tensors.
        feed_model_data: Function to feed input tensors to the model.
        metric_name: One of 'val_f1', 'val_accuracy', or 'val_min_per_class_f1'.
        mtll_model: Optional multi-task loss learner.
        num_main_losses: Number of main losses.
        **kwargs: Extra parameters passed to feed_model_data.
    Returns:
        Tuple of (total_val_loss, val_losses_list, total_raw_loss, raw_losses_list, metric_value).
    """
    model.eval()
    if mtll_model is not None:
        mtll_model.eval()
    test_raw_losses = None
    test_losses = None
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for dataset in val_loader:
            data, target = fetch_model_data(dataset, device=device)
            output = feed_model_data(model, data, **kwargs)
            raw_losses = criterion(output, target, reduction='mean')
            if mtll_model is not None:
                test_raw_losses = _collect_losses(raw_losses, test_raw_losses)
                losses = mtll_model(raw_losses)
            else:
                losses = raw_losses
            test_losses = _collect_losses(losses, test_losses)
            # Collect predictions and targets for metric computation
            if num_main_losses is not None:
                metric_output = output[-num_main_losses:]
                metric_target = target[-num_main_losses:]
            else:
                metric_output = output
                metric_target = target
            preds = torch.argmax(metric_output[0], dim=1).cpu().numpy()
            t = metric_target[0].squeeze(-1)
            if t.ndim > 1:
                t = t.mode(dim=1).values
            t = t.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(t)

    assert test_losses is not None
    num_main_losses_resolved = num_main_losses if num_main_losses is not None else len(test_losses)
    test_losses = [test_loss / len(val_loader) for test_loss in test_losses]
    total_test_loss = sum(test_losses[-num_main_losses_resolved:])
    test_raw_losses = [rl / len(val_loader) for rl in test_raw_losses] if test_raw_losses is not None else None
    total_test_raw_loss = sum(test_raw_losses[-num_main_losses_resolved:]) if test_raw_losses is not None else None

    # Compute metric
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    # Always compute per-class F1 (used by quality gate)
    per_class_f1s = f1_score(all_targets, all_preds, average=None, zero_division="warn")  # type: ignore[arg-type]
    min_per_class_f1 = float(np.min(per_class_f1s))
    if metric_name == "val_f1":
        metric_value = float(f1_score(all_targets, all_preds, average="weighted", zero_division="warn"))
    elif metric_name == "val_accuracy":
        metric_value = float(accuracy_score(all_targets, all_preds))
    elif metric_name == "val_min_per_class_f1":
        metric_value = min_per_class_f1
    else:
        raise ValueError(f"Unknown metric_name: {metric_name}")

    return total_test_loss, test_losses, total_test_raw_loss, test_raw_losses, metric_value, min_per_class_f1


def train(model, train_loader, optimizer, criterion, epochs, device, loss_names, clip_gradient_at=0.0,
          fetch_model_data=single_input_single_output, feed_model_data=basic_forward,
          val_loader=None, initial_epoch=1, mtll_model=None, print_raw_losses=False,
          num_main_losses=None, early_stopping_patience=None, min_epochs=10,
          early_stopping_metric="val_loss", min_per_class_f1_thresh=None,
          lr_increase_patience=None, lr_increase_factor=1.5, scheduler=None, **kwargs):
    """General training function to train a PyTorch model.

    If validation data is not given, the returned checkpoint is the one obtained after training the model for the
    specified number of epochs, regardless of the final training loss. If validation data is given, the checkpoint
    returned is the one with the lowest validation loss, which could have been obtained in some epoch before the
    last one.
    Arg(s):
        model - PyTorch model.
        train_loader - Batch generator for model training.
        optimizer - Model optimizer.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        epochs - Maximum number of epochs for model training.
        device - Which device to use for model training. Either cuda or cpu.
        loss_names - Names for the individual losses output by criterion.
        clip_gradient_at - If nonzero clips the norm of the gradient vector at the specified value. The gradient
            vector is a vector obtained by concatenating all parameters of the model.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        val_loader - Batch generator for model validation.
        early_stopping_patience - Stop training if validation metric doesn't improve for this many epochs.
            If None, no early stopping is applied (fixed epochs). Requires val_loader to be provided.
        min_epochs - Minimum number of epochs before early stopping can activate. Prevents stopping
            during noisy early training. Default: 10.
        early_stopping_metric - Metric to use for early stopping. One of:
            'val_loss' (default): Use validation loss (lower is better).
            'val_f1': Use weighted F1 score on validation set (higher is better).
            'val_accuracy': Use accuracy on validation set (higher is better).
            'val_min_per_class_f1': Use minimum per-class F1 score (higher is better).
        min_per_class_f1_thresh - Quality gate: minimum threshold (0.0-1.0) for per-class F1.
            When set, training will NOT stop (even if patience is exhausted) until the minimum
            per-class F1 on the validation set reaches this threshold. Works with any
            early_stopping_metric. If None, no quality gate is applied.
        lr_increase_patience - If set, increase learning rate by lr_increase_factor when
            validation metric hasn't improved for this many epochs. Helps escape local minima
            when the model is stuck. If None, no LR adjustment is applied.
        lr_increase_factor - Factor to multiply LR by when stuck (default: 1.5).
        **kwargs - Any extra parameters to be passed during training.
    Returns:
        A dictionary containing the history of train losses, the model's weights and associated epoch, and if
        val_loader is specified, the history of validation losses as well.
    """
    checkpoint_name = kwargs.get('checkpoint_name')
    tensorboard_log_dir = kwargs.get('tensorboard_log_dir')
    writer = None
    if tensorboard_log_dir is not None and checkpoint_name is not None:
        writer = SummaryWriter(os.path.join(tensorboard_log_dir, 'runs', checkpoint_name))
    checkpoint = {}
    train_losses, val_losses, train_raw_losses, val_raw_losses = [], [], [], []
    val_loss = float('Inf')
    epochs_without_improvement = 0
    stopped_early = False
    stopped_epoch = epochs + initial_epoch - 1

    # Configure early stopping metric
    _valid_metrics = ('val_loss', 'val_f1', 'val_accuracy', 'val_min_per_class_f1')
    if early_stopping_metric not in _valid_metrics:
        raise ValueError(f"early_stopping_metric must be one of {_valid_metrics}, got '{early_stopping_metric}'")
    # For val_f1, val_accuracy, and val_min_per_class_f1, higher is better; for val_loss, lower is better
    _metric_is_higher_better = early_stopping_metric in ('val_f1', 'val_accuracy', 'val_min_per_class_f1')
    _best_metric_value = -float('Inf') if _metric_is_higher_better else float('Inf')
    _metric_history = []  # Track metric values for logging
    _current_min_per_class_f1 = None  # Track min per-class F1 for quality gate

    # Validate early stopping configuration
    if early_stopping_patience is not None and val_loader is None:
        print('WARNING: early_stopping_patience is set but no val_loader provided. Early stopping will be disabled.')
        early_stopping_patience = None

    if early_stopping_patience is not None:
        print(f'Early stopping enabled: patience={early_stopping_patience}, min_epochs={min_epochs}, metric={early_stopping_metric}')
    if min_per_class_f1_thresh is not None:
        print(f'Quality gate enabled: min_per_class_f1_thresh={min_per_class_f1_thresh:.2%}')
    if lr_increase_patience is not None:
        print(f'LR increase enabled: lr_increase_patience={lr_increase_patience}, lr_increase_factor={lr_increase_factor}')
    if scheduler is not None:
        print(f'LR scheduler enabled: {scheduler.__class__.__name__}')

    for epoch in range(initial_epoch, epochs + initial_epoch):
        # Train
        print(f'\nEpoch: [{epoch:4d}/{epochs + initial_epoch - 1:4d}]')
        train_single_epoch(model, data_loader=train_loader, optimizer=optimizer, criterion=criterion,
                           device=device, loss_names=loss_names, clip_gradient_at=clip_gradient_at,
                           fetch_model_data=fetch_model_data, feed_model_data=feed_model_data, log_interval=25,
                           mtll_model=mtll_model, num_main_losses=num_main_losses, **kwargs)
        current_train_loss, current_train_losses, current_train_raw_loss, current_train_raw_losses = \
            test(model, data_loader=train_loader, criterion=criterion,
                 device=device, loss_names=loss_names, fetch_model_data=fetch_model_data,
                 feed_model_data=feed_model_data, test_set_name='Train', mtll_model=mtll_model,
                 print_raw_losses=print_raw_losses, num_main_losses=num_main_losses, **kwargs)
        train_losses.append([current_train_loss, current_train_losses])
        if mtll_model is not None:
            train_raw_losses.append([current_train_raw_loss, current_train_raw_losses])
        if writer is not None:
            base_str = 'Loss/train_mtll/' if mtll_model is not None else 'Loss/train/'
            for loss_name, loss in zip(loss_names, current_train_losses):
                writer.add_scalar(base_str + loss_name, loss, epoch)
            writer.add_scalar(base_str + 'total', current_train_loss, epoch)
            if mtll_model is not None:
                loss_weights = mtll_model.get_weights()
                assert current_train_raw_losses is not None
                for loss_name, raw_loss, loss_weight in zip(loss_names, current_train_raw_losses, loss_weights):
                    writer.add_scalar(f'Loss/train/{loss_name}', raw_loss, epoch)
                    if loss_weight is not None:
                        writer.add_scalar(f'Loss/mtll_weight/{loss_name}', loss_weight, epoch)
                writer.add_scalar('Loss/train/total', current_train_raw_loss, epoch)
        # Validate
        if val_loader is not None:
            if early_stopping_metric == 'val_loss':
                # Single pass: just compute loss (original behavior)
                current_val_loss, current_val_losses, current_val_raw_loss, current_val_raw_losses = \
                    test(model, data_loader=val_loader, criterion=criterion,
                         device=device, loss_names=loss_names, fetch_model_data=fetch_model_data,
                         feed_model_data=feed_model_data, test_set_name='Validation', mtll_model=mtll_model,
                         print_raw_losses=print_raw_losses, num_main_losses=num_main_losses)
                current_metric = current_val_loss
            else:
                # Single pass: compute loss + metric together (avoid two separate passes)
                current_val_loss, current_val_losses, current_val_raw_loss, current_val_raw_losses, current_metric, current_min_per_class_f1 = \
                    _validate_with_metric(
                        model, val_loader, criterion, device, loss_names,
                        fetch_model_data, feed_model_data,
                        metric_name=early_stopping_metric,
                        mtll_model=mtll_model, num_main_losses=num_main_losses, **kwargs,
                    )
                _current_min_per_class_f1 = current_min_per_class_f1
                print(f'  {early_stopping_metric}: {current_metric:.4f}')
                if min_per_class_f1_thresh is not None:
                    print(f'  min_per_class_f1: {current_min_per_class_f1:.4f} (thresh={min_per_class_f1_thresh:.4f})')
            val_losses.append([current_val_loss, current_val_losses])
            if mtll_model is not None:
                val_raw_losses.append([current_val_raw_loss, current_val_raw_losses])
            if writer is not None:
                base_str = 'Loss/val_mtll/' if mtll_model is not None else 'Loss/val/'
                for loss_name, loss in zip(loss_names, current_val_losses):
                    writer.add_scalar(base_str + loss_name, loss, epoch)
                writer.add_scalar(base_str + 'total', current_val_loss, epoch)
                if mtll_model is not None:
                    assert current_val_raw_losses is not None
                    for loss_name, raw_loss in zip(loss_names, current_val_raw_losses):
                        writer.add_scalar('Loss/val/' + loss_name, raw_loss, epoch)
                    writer.add_scalar('Loss/val/total', current_val_raw_loss, epoch)
            _metric_history.append(current_metric)

            # Check if metric improved
            _metric_improved = (
                current_metric > _best_metric_value if _metric_is_higher_better
                else current_metric < _best_metric_value
            )
            if _metric_improved:
                _best_metric_value = current_metric
                val_loss = current_val_loss  # Always track val_loss for reference
                epochs_without_improvement = 0
                checkpoint['epoch'] = epoch
                checkpoint['model_state_dict'] = model.state_dict()
                if mtll_model is not None:
                    checkpoint['mtll_model_state_dict'] = mtll_model.state_dict()
            else:
                epochs_without_improvement += 1
                # LR increase: when stuck, increase LR to escape local minima
                if (lr_increase_patience is not None and
                        epochs_without_improvement > 0 and
                        epochs_without_improvement % lr_increase_patience == 0):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_increase_factor
                    print(f'  LR increased to {optimizer.param_groups[0]["lr"]:.6f} (stuck for {epochs_without_improvement} epochs)')

            # Early stopping check (only after min_epochs)
            if (early_stopping_patience is not None and
                epoch >= min_epochs and
                epochs_without_improvement >= early_stopping_patience):
                # Quality gate: only stop if min per-class F1 threshold is met
                # Check the BEST value achieved, not the current value
                if (min_per_class_f1_thresh is not None and
                        _current_min_per_class_f1 is not None and
                        _best_metric_value < min_per_class_f1_thresh):
                    print(f'  WARNING: Patience exhausted but quality gate NOT met '
                          f'(best min_per_class_f1={_best_metric_value:.4f} < {min_per_class_f1_thresh:.4f}). '
                          f'Continuing training...')
                    epochs_without_improvement = 0  # Reset to keep training
                else:
                    print(f'\nEarly stopping triggered at epoch {epoch}')
                    print(f'  No improvement in {early_stopping_metric} for {epochs_without_improvement} epochs (patience={early_stopping_patience})')
                    print(f'  Best {early_stopping_metric}: {_best_metric_value:.4f}')
                    if _current_min_per_class_f1 is not None:
                        print(f'  Final min per-class F1: {_current_min_per_class_f1:.4f}')
                    stopped_early = True
                    stopped_epoch = epoch
                    break

            # Step LR scheduler (end of epoch, sets LR for next epoch)
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                if writer is not None:
                    writer.add_scalar('LR', current_lr, epoch)

            # Reset counter during min_epochs window so early stopping only counts after min_epochs
            if epoch < min_epochs:
                epochs_without_improvement = 0
        else:
            # No val_loader: still step scheduler
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                if writer is not None:
                    writer.add_scalar('LR', current_lr, epoch)

            checkpoint['epoch'] = epoch
            checkpoint['model_state_dict'] = model.state_dict()
            if mtll_model is not None:
                checkpoint['mtll_model_state_dict'] = mtll_model.state_dict()
    print(f'Lowest val_loss is {val_loss:.4f}')
    if early_stopping_metric != 'val_loss':
        print(f'Best {early_stopping_metric} is {_best_metric_value:.4f}')
    checkpoint['train_losses'] = train_losses
    checkpoint['val_losses'] = val_losses
    checkpoint['train_raw_losses'] = train_raw_losses
    checkpoint['val_raw_losses'] = val_raw_losses
    checkpoint['stopped_early'] = stopped_early
    checkpoint['stopped_epoch'] = stopped_epoch
    checkpoint['epochs_without_improvement'] = epochs_without_improvement
    checkpoint['early_stopping_metric'] = early_stopping_metric
    checkpoint['best_metric_value'] = _best_metric_value
    checkpoint['metric_history'] = _metric_history
    checkpoint['min_per_class_f1_thresh'] = min_per_class_f1_thresh
    checkpoint['lr_increase_patience'] = lr_increase_patience
    checkpoint['lr_increase_factor'] = lr_increase_factor
    if early_stopping_metric == 'val_min_per_class_f1':
        checkpoint['min_per_class_f1_history'] = list(_metric_history)
    else:
        checkpoint['min_per_class_f1_history'] = []
    if _current_min_per_class_f1 is not None:
        checkpoint['final_min_per_class_f1'] = _current_min_per_class_f1
    if writer is not None:
        writer.close()
    return checkpoint


def train_single_epoch(model, data_loader, optimizer, criterion, device, loss_names, clip_gradient_at=0.0,
                       fetch_model_data=single_input_single_output, feed_model_data=basic_forward,
                       log_interval=25, mtll_model=None, num_main_losses=None, **kwargs):
    """General training function to train a PyTorch model for a single epoch.

    Arg(s):
        model - PyTorch model.
        data_loader - Batch generator for model training.
        optimizer - Model optimizer.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        device - Which device to use for model training. Either cuda or cpu.
        loss_names - Names for the individual losses output by criterion.
        clip_gradient_at - If nonzero clips the norm of the gradient vector at the specified value. The gradient
            vector is a vector obtained by concatenating all parameters of the model.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        log_interval - Print training statistics every log_interval batches.
        **kwargs - Any extra parameter that needs to be passed to the feed_model_data of a model.
    """
    model.train()
    if mtll_model is not None:
        mtll_model.train()
    num_examples = len(data_loader.dataset)
    for batch_idx, dataset in enumerate(data_loader):
        data, target = fetch_model_data(dataset, device=device)
        optimizer.zero_grad()
        output = feed_model_data(model, data, **kwargs)
        losses = criterion(output, target, reduction='mean')
        if mtll_model is not None:
            losses = mtll_model(losses)
        loss = sum(losses)
        loss.backward()  # type: ignore[union-attr]
        if clip_gradient_at:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_at)
        optimizer.step()
        log_now, is_last_batch = (batch_idx % log_interval) == 0, batch_idx == (len(data_loader) - 1)
        if log_now or is_last_batch:
            num_main_losses = num_main_losses if num_main_losses is not None else len(losses)
            loss = sum(losses[-num_main_losses:])
            batch_initial_example_idx = min((batch_idx + 1) * data_loader.batch_size, num_examples)
            epoch_progress = 100 * (batch_idx + 1) / len(data_loader)
            print(f'(Train) Batch [{batch_initial_example_idx:6d}/{num_examples:6d} ({epoch_progress:3.0f}%)] ',
                  f'Loss: {loss.item(): 8.4f}', end='')  # type: ignore[union-attr]
            for loss_name, single_loss in zip(loss_names, losses):
                print(f'  {loss_name}: {single_loss: 6.4f}', end='')
            print()


def test(model, data_loader, criterion, device, loss_names, fetch_model_data=single_input_single_output,
         feed_model_data=basic_forward, test_set_name='Test', mtll_model=None,
         print_raw_losses=False, num_main_losses=None, **kwargs):
    """General testing function to test a PyTorch model.

    Arg(s):
        model - PyTorch model.
        data_loader - Batch generator for model testing.
        criterion - Specific loss function for the given model. This function receives as input the output of
            model and the ground-truth target, and returns a list of batch losses (for multi-loss models). Even if
            the model has a single loss, the return value of criterion must be a list containing this single loss.
        device - Which device to use for model testing. Either cuda or cpu.
        loss_names - Names for the individual losses output by criterion.
        fetch_model_data - Function to fetch the input and output tensors for the model.
        feed_model_data - Function to feed the input tensors to the model.
        test_set_name - Optional name given to the set being evaluated. Useful for logging purposes.
        num_main_losses - The final test loss is the sum of all non-auxiliary losses. Auxiliary losses should be in
            the beginning of the output list.
        **kwargs - Any extra parameters that need to be passed to the feed_model_data function.
    Returns:
        The model loss.
    """
    model.eval()
    if mtll_model is not None:
        mtll_model.eval()
    test_raw_losses = None
    test_losses = None
    with torch.no_grad():
        for dataset in data_loader:
            data, target = fetch_model_data(dataset, device=device)
            output = feed_model_data(model, data, **kwargs)
            raw_losses = criterion(output, target, reduction='mean')
            if mtll_model is not None:
                test_raw_losses = _collect_losses(raw_losses, test_raw_losses)
                losses = mtll_model(raw_losses)
            else:
                losses = raw_losses
            test_losses = _collect_losses(losses, test_losses)
    assert test_losses is not None
    num_main_losses = num_main_losses if num_main_losses is not None else len(test_losses)
    test_losses = [test_loss / len(data_loader) for test_loss in test_losses]
    total_test_loss = sum(test_losses[-num_main_losses:])

    test_set_name = f'({test_set_name})'
    print(f'{test_set_name:>12} Loss: {total_test_loss: 7.4f}', end='')
    for loss_name, loss in zip(loss_names, test_losses):
        print(f'   {loss_name}: {loss: 6.4f}', end='')
    print()
    total_test_raw_loss = None
    if test_raw_losses is not None:
        test_raw_losses = [test_raw_loss / len(data_loader) for test_raw_loss in test_raw_losses]
        total_test_raw_loss = sum(test_raw_losses[-num_main_losses:])
        if print_raw_losses:
            print(f'{test_set_name:>12} Loss: {total_test_raw_loss: 7.4f}', end='')
            for loss_name, raw_loss in zip(loss_names, test_raw_losses):
                print(f'   {loss_name}: {raw_loss: 6.4f}', end='')
            print()
    return total_test_loss, test_losses, total_test_raw_loss, test_raw_losses


def _collect_losses(output_losses: list, losses: Optional[list] = None):  # type: ignore[valid-type]
    if losses is None:
        return [output_loss.item() for output_loss in output_losses]
    return [loss + output_loss.item() for loss, output_loss in zip(losses, output_losses)]


def save_checkpoint(log_dir, checkpoint: dict, checkpoint_name: Optional[str] = None, include_timestamp: bool = False):  # type: ignore[valid-type]
    """Save PyTorch model checkpoint.

    Arg(s):
        log_dir - Directory to save checkpoint file. It must already exist.
        checkpoint - A dictionary containing the model checkpoint and other metadata such as data scalers and
            model creation arguments.
        checkpoint_name - If given, use that as the file name to save. Otherwise, the file name is 'checkpoint'. A
            '.tar' is appended to checkpoint_name, so there is no need to include it in the passed checkpoint name.
        include_timestamp - Whether to prepend the timestamp to the checkpoint name or not.
    """
    file_save_name = checkpoint_name if checkpoint_name is not None else 'checkpoint'
    if include_timestamp:
        time_now = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        file_save_name = time_now + '_' + file_save_name
    file_save_name += '.tar'
    file_save_path = os.path.join(log_dir, file_save_name)
    torch.save(checkpoint, file_save_path)
    print(f'log files written to {file_save_path}')


def numpy_to_torch(*arrays, device='cpu'):
    """Convert any number of numpy arrays to PyTorch tensors."""
    return [torch.from_numpy(array).to(device) for array in arrays]
