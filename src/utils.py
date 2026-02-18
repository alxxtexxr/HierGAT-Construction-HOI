import re
from pathlib import Path

import numpy as np
import pandas as pd

def printh(h, n=128):
    print("=" * n)
    print(h)
    print("=" * n)
    
def get_feature_dirs_df(feature_dirs, action_classes):
    feature_dir_dicts = []
    for dirs in feature_dirs:
        for dir in dirs.iterdir():
            action_label_str = str(dir).rsplit('_action_')[-1]
            is_interpolated = 0
            if '_interp' in action_label_str:
                action_label_str = action_label_str.replace('_interp', '')
                is_interpolated = 1
            action_label = int(action_label_str)
            action_class = action_classes[action_label]
            if 'new_action_classes' in globals():
                try:
                    action_label = new_action_classes.index(action_class) # type: ignore
                except ValueError:
                    action_label = -1
            feature_dir_dicts.append({
                'base_dir': str(dirs),
                'dir': str(dir),
                'action_label': action_label,
                # 'new_action_label': new_action_label,
                'is_interpolated': is_interpolated,
            })
    feature_dirs_df = pd.DataFrame(feature_dir_dicts)
    feature_dirs_df = feature_dirs_df[feature_dirs_df['action_label'] != -1] # remove rows where action_label == -1
    return feature_dirs_df

def create_data_df(feature_dirs, action_classes, new_action_classes=None, 
                   downsampling: int = 1, test_data: bool = False):
    base_dirs = []
    dirs = []
    start_fids = []
    end_fids = []
    human_features_list = []
    human_boxes_list = []
    human_poses_list = []
    object_features_list = []
    object_boxes_list = []
    gt_list = [] if not test_data else None
    action_labels = [] if not test_data else None
    xs_steps = []

    for feature_dir in feature_dirs:
        feature_dir = Path(feature_dir)
        
        # Store directories
        base_dirs.append(str(feature_dir.parent.parent))
        dirs.append(str(feature_dir))
        
        # Store start and end FIDs
        match = re.search(r"range_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)", str(feature_dir))
        start_fid, end_fid = map(float, match.groups())
        start_fids.append(start_fid)
        end_fids.append(end_fid)
        
        # Load and store human (subject) features
        subject_visual_features = np.load(feature_dir / 'subject_visual_features.npy')
        subject_boxes = np.load(feature_dir / 'subject_boxes.npy')
        # subject_poses = np.zeros((subject_visual_features.shape[0], 17, 2))
        subject_poses = np.load(feature_dir / 'subject_poses.npy')
        
        human_features_list.append([subject_visual_features])
        human_boxes_list.append([subject_boxes])
        human_poses_list.append([subject_poses])
        
        # Load and store object features
        object_visual_features = np.load(feature_dir / 'object_visual_features.npy')
        object_boxes = np.load(feature_dir / 'object_boxes.npy')
        
        object_features_list.append(object_visual_features[:, np.newaxis, :])
        object_boxes_list.append(object_boxes[:, np.newaxis, :])
        
        # Extract and store ground-truth action label
        if not test_data:
            action_label_str = str(feature_dir).split('_action_')[-1]
            if '_' in action_label_str:
                action_label_str = action_label_str.split('_')[0]
            action_label = int(action_label_str)
            if isinstance(new_action_classes, (list, tuple)):
                action_class = action_classes[action_label]
                action_label = new_action_classes.index(action_class)
            seq_len = subject_visual_features.shape[0]
            gt_list.append({
                'Human1': [action_label] * seq_len,
            })
            action_labels.append(action_label)
        
        # Store number of steps
        num_steps = len(subject_visual_features[downsampling - 1::downsampling])
        xs_steps.append(num_steps)

    xs_steps = np.array(xs_steps, dtype=np.float32)

    data_df = pd.DataFrame({
        'human_features': human_features_list,
        'human_boxes': human_boxes_list,
        'human_poses': human_poses_list,
        'object_features': object_features_list,
        'object_boxes': object_boxes_list,
        'gt': gt_list,
        'xs_step': xs_steps,
        'base_dir': base_dirs,
        'dir': dirs,
        'start_fid': start_fids,
        'end_fid': end_fids,
        'action_label': action_labels,
    })
    data_df['xs_step'] = data_df['xs_step'].astype(int)
    data_df['start_fid'] = data_df['start_fid'].astype(int)
    data_df['end_fid'] = data_df['end_fid'].astype(int)
    return data_df