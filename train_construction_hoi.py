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
from vhoi.data_loading_custom import (
    create_data,
    create_data_loader
)
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
from src.constants import ACTION_CLASSES, NEW_ACTION_CLASSES, VIS_ACTION_CLASSES, FEATURE_DIRS
from src.utils import printh, get_feature_dirs_df, create_data_df

torch.multiprocessing.set_sharing_strategy('file_system') # Avoid shared memory crashes

def main(
    config_path = 'conf/config_construction_hoi.yaml',
    seed = 42,
    k = 10,
    num_epochs = None, # Default: number of epochs in configuration file
):
    # Set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
    
    torch.set_num_threads(cfg.resources.num_threads)
    model_name, model_input_type = cfg.models.metadata.model_name, cfg.models.metadata.input_type
    batch_size, val_fraction = cfg.models.optimization.batch_size, cfg.models.optimization.val_fraction
    misc_dict = cfg.get('misc', default_value={})
    sigma = misc_dict.get('segmentation_loss', {}).get('sigma', 0.0)
    scaling_strategy = cfg.data.scaling_strategy
    downsampling = cfg.data.downsampling
    
    # Determine test feature directories for k-fold
    feature_dirs_df = get_feature_dirs_df(FEATURE_DIRS, ACTION_CLASSES, 
                                          NEW_ACTION_CLASSES if 'NEW_ACTION_CLASSES' in globals() else None)
    feature_dir_counts = feature_dirs_df['base_dir'].value_counts().sort_values() # Count occurrences of each feature directory 
                                                                                  # and sort from lowest to highest
    kfold_test_feature_dirs = feature_dir_counts.index[:k]
    
    # Create evaluation directory
    run_name = (
        f"hs{cfg.models.parameters.hidden_size}_"
        f"e{cfg.models.optimization.epochs}_"
        f"bs{cfg.models.optimization.batch_size}_"
        f"lr{cfg.models.optimization.learning_rate}_"
        f"k{k}_"
        f"{datetime.now().strftime('%y%m%d%H%M%S')}"
    )
    eval_dir = f'{os.getcwd()}/eval/{run_name}'
    os.makedirs(eval_dir, exist_ok=True)
    
    # Iterate through k-fold
    acc_list, f1_list, precision_list, recall_list, cm_list = [], [], [], [], []
    for fold, test_feature_dir in enumerate(kfold_test_feature_dirs):
        printh(f"Fold {fold+1}: {test_feature_dir}", 128)
        
        # Create log directory
        root_log_dir = f'{os.getcwd()}/outputs_hiergat/construction_hoi'
        fold_str = str(fold+1).zfill(len(str(k)))
        checkpoint_name = f'{run_name}_fold{fold_str}'
        log_dir = f"{root_log_dir}/{checkpoint_name}"
        os.makedirs(log_dir, exist_ok=True)
        
        print("Log directory:", log_dir)
        print()
        
        # Split train and test feature directories
        printh("Train-Test Splitting", 96)
        
        test_feature_dirs_df = feature_dirs_df[feature_dirs_df['base_dir'] == test_feature_dir]
        train_feature_dirs_df = feature_dirs_df[feature_dirs_df['base_dir'] != test_feature_dir]
        
        print("Training action label counts:")
        print(train_feature_dirs_df['action_label'].value_counts().sort_index())
        print()
        print("Test action label counts:")
        print(test_feature_dirs_df['action_label'].value_counts().sort_index())
        print()
        
        # Create data dataframe
        df = create_data_df(train_feature_dirs_df['dir'].tolist(), ACTION_CLASSES, 
                            NEW_ACTION_CLASSES if 'NEW_ACTION_CLASSES' in globals() else None)
        
        # Split train and validation feature data
        printh("Train-Validation Splitting", 96)
        
        for base_dir, group_df in df.groupby('base_dir'):
            printh(f"Feature Directory: {base_dir}", 64)
            
            # group_df['split'] = 'train' # OLD
            df.loc[df['base_dir'] == base_dir, 'split'] = 'train'
            group_df = group_df.sort_values('start_fid')
            
            min_fid = 0
            max_fid = group_df['end_fid'].max()
            duration = max_fid - min_fid
            val_duration = int(duration * val_fraction)
            
            # Determine validation and testing FID ranges
            assert duration//2 >= val_duration+val_duration, f"Duration ({duration}) should be at least twice the validation duration ({val_duration})!"
            # val_start_fid = random.randrange(0, duration//2) # Sample a random FID from the start to the middle
            val_start_fid   = random.randrange(duration//2, duration-val_duration-val_duration) # Sample a random FID from the middle to the end
            val_end_fid     = val_start_fid + val_duration - 1
            test_start_fid  = val_start_fid + val_duration
            test_end_fid    = test_start_fid + val_duration - 1
            
            print(f"Validation FID range: {val_start_fid}-{val_end_fid}, length: {val_end_fid-val_start_fid+1}")
            print(f"Testing FID range: {test_start_fid}-{test_end_fid}, length: {test_end_fid-test_start_fid+1}")
            print()
            
            # Swap validation and testing FID ranges if validation is larger than testing
            base_dir_cond = df['base_dir'] == base_dir
            val_len = len(df.loc[base_dir_cond & (df['start_fid'] >= val_start_fid) & (df['end_fid'] <= val_end_fid)])
            test_len = len(df.loc[base_dir_cond & (df['start_fid'] >= test_start_fid) & (df['end_fid'] <= test_end_fid)])
            if val_len > test_len:
                tmp_start_fid  = test_start_fid
                tmp_end_fid    = test_end_fid
                test_start_fid = val_start_fid
                test_end_fid   = val_end_fid
                val_start_fid  = tmp_start_fid
                val_end_fid    = tmp_end_fid
                
            # Assign splits
            df.loc[base_dir_cond & (df['start_fid'] >= val_start_fid) & (df['end_fid'] <= val_end_fid), 'split'] = 'val'
            # df.loc[base_dir_cond & (df['start_fid'] >= test_start_fid) & (df['end_fid'] <= test_end_fid), 'split'] = 'test'
            df.loc[base_dir_cond & (df['split'].isna()), 'split'] = 'train'
            
            train_group_df = df[base_dir_cond & (df['split'] == 'train')]
            val_group_df = df[base_dir_cond & (df['split'] == 'val')]
            test_group_df = df[base_dir_cond & (df['split'] == 'test')]
            
            try:
                assert (len(test_group_df) > len(val_group_df) or len(test_group_df) == 0) and \
                        len(df[base_dir_cond]) == len(train_group_df)+len(val_group_df)+len(test_group_df)
            except:
                import pdb; pdb.set_trace()
                
            print("Training action label info:")
            print(train_group_df['action_label'].value_counts())
            print("Total:", len(train_group_df))
            print()
            print("Validation action label info:")
            print(val_group_df['action_label'].value_counts())
            print("Total:", len(val_group_df))
            print()
            print("Testing action label info:")
            print(test_group_df['action_label'].value_counts())
            print("Total:", len(test_group_df))
            print()
        
        # Create train, validation, and test dataframes
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']

        train_size = len(train_df)
        val_size = len(val_df)
        test_size = len(test_df)
        all_size = len(df)

        print(f"Total train data: {train_size} ({train_size/all_size*100:.2f}%)")
        print(f"Total val data: {val_size} ({val_size/all_size*100:.2f}%)")
        print(f"Total test data: {test_size} ({test_size/all_size*100:.2f}%)")
        print()
        
        data_cols  = ['human_features', 'human_boxes', 'human_poses', 'object_features', 'object_boxes', 'gt', 'xs_step']
        train_data = train_df[data_cols].to_numpy().T.tolist()
        val_data   = val_df[data_cols].to_numpy().T.tolist()
        test_data = create_data(test_feature_dirs_df['dir'].tolist(), ACTION_CLASSES, 
                                NEW_ACTION_CLASSES if 'NEW_ACTION_CLASSES' in globals() else None)
        
        # Create data loaders
        train_loader, scalers, _ = create_data_loader(
            *train_data, 
            model_name, 
            batch_size=batch_size, 
            shuffle=True,
            scaling_strategy=scaling_strategy, 
            sigma=sigma,
            downsampling=downsampling,
        )
        val_loader, _, _ = create_data_loader(
            *val_data, 
            model_name, 
            batch_size=len(val_data[0]),
            shuffle=False, 
            scalers=scalers, 
            sigma=sigma, 
            downsampling=downsampling,
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
        input_size = input_size_from_data_loader(train_loader, model_name, model_input_type)
        data_info = {'input_size': input_size}
        
        # Create model
        model_creation_args = cfg.models.parameters
        model_creation_args = {**data_info, **model_creation_args}
        dataset_name = cfg.data.name
        num_classes = len(NEW_ACTION_CLASSES)
        model_creation_args['num_classes'] = (num_classes, None)
        device = 'cuda' if torch.cuda.is_available() and cfg.resources.use_gpu else 'cpu'
        model = TGGCN(feat_dim=1024, **model_creation_args).to(device)
        if misc_dict.get('pretrained', False) and misc_dict.get('pretrained_path') is not None:
            state_dict = load_model_weights(misc_dict['pretrained_path'])
            model.load_state_dict(state_dict, strict=False)
        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=cfg.models.optimization.learning_rate)
        criterion, loss_names = select_loss(model_name, model_input_type, dataset_name, cfg=cfg)
        mtll_model = None
        if misc_dict.get('multi_task_loss_learner', False):
            loss_types = select_loss_types(model_name, dataset_name, cfg=cfg)
            mask = select_loss_learning_mask(model_name, dataset_name, cfg=cfg)
            mtll_model = MultiTaskLossLearner(loss_types=loss_types, mask=mask).to(device)
            optimizer.add_param_group({'params': mtll_model.parameters()})
        # Some config + model training
        tensorboard_log_dir = root_log_dir # OLD: cfg.models.logging.root_log_dir
        # checkpoint_name = cfg.models.logging.checkpoint_name
        fetch_model_data = select_model_data_fetcher(model_name, model_input_type,
                                                    dataset_name=dataset_name, **{**misc_dict, **cfg.models.parameters.__dict__})
        feed_model_data = select_model_data_feeder(model_name, model_input_type, dataset_name=dataset_name, **misc_dict)
        num_main_losses = decide_num_main_losses(model_name, dataset_name, {**misc_dict, **cfg.models.parameters.__dict__})
        
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
            fetch_model_data=fetch_model_data, feed_model_data=feed_model_data,
            val_loader=val_loader, 
            mtll_model=mtll_model, 
            num_main_losses=num_main_losses,
            tensorboard_log_dir=tensorboard_log_dir, 
            checkpoint_name=checkpoint_name,
        )
        
        print()
        
        # Logging
        # if cfg.models.logging.log_dir is not None: # OLD
        if log_dir is not None:
            # log_dir = cfg.models.logging.log_dir
            checkpoint['scalers'] = scalers
            save_checkpoint(log_dir, checkpoint, checkpoint_name=checkpoint_name, include_timestamp=False)

        # Evaluation
        printh("Evaluation", 96)
        print("Selected test video ID:", Path(test_feature_dir).parent.stem[:5])
        print()

        model.eval()
        
        inspect_model = False
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
                for i, (out, tgt) in enumerate(zip(output, target)):
                    if out.ndim != 4:
                        raise RuntimeError(f'Number of dimensions for output is {out.ndim}')
                    out = torch.repeat_interleave(out, repeats=downsampling, dim=-2)
                    out = match_shape(out, tgt)
                    output[i] = out
                if inspect_model:
                    a_target = target[0]
                    attention_scores = [torch.repeat_interleave(att_score, repeats=downsampling, dim=-2)
                                        for att_score in attention_scores]
                    attention_scores = [match_att_shape(att_score, a_target) for att_score in attention_scores]
                    attentions.append(attention_scores)
            outputs += output
            targets += target

        y_pred = torch.argmax(outputs[0], dim=1).cpu().numpy()
        y_true = targets[0].squeeze(-1).mode(dim=1).values.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        print("Accuracy:", acc)
        print("F1 score:", f1)
        print("Precision:", precision)
        print("Recall:", recall)
        
        # Plot confusion matrix
        # ticklabels = action_classes
        ticklabels = NEW_ACTION_CLASSES
        if 'new_action_classes' in globals() and isinstance(new_action_classes, (list, tuple)): # type: ignore
            y_true = [ticklabels.index(new_action_classes[y]) for y in y_true] # type: ignore
            y_pred = [ticklabels.index(new_action_classes[y]) for y in y_pred] # type: ignore
            
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ticklabels))))
        
        print("Confusion matrix:")
        print(cm)
        print()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                    xticklabels=VIS_ACTION_CLASSES, yticklabels=VIS_ACTION_CLASSES)
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")
        
        # Save confusion matrix
        cm_dir = f'{eval_dir}/cm'
        os.makedirs(cm_dir, exist_ok=True)
        plt.savefig(f'{cm_dir}/fold{fold_str}.png')
        plt.close()
        
        # Cleanup (VERY IMPORTANT)
        del model, optimizer
        del train_loader, val_loader, test_loader
        del train_data, val_data, test_data
        del df, train_df, val_df, test_df
        del outputs, targets
        torch.cuda.empty_cache()
        gc.collect()
        
        # Store metrics
        acc_list.append(acc)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        cm_list.append(cm)
    
    # Save metrics
    metrics = {
        'acc': acc_list,
        'f1': f1_list,
        'precision': precision_list,
        'recall': recall_list,
        'cm': cm_list
    }
    
    with open(f'{eval_dir}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    fire.Fire(main)