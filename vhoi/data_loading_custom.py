import itertools
from typing import Optional
from pathlib import Path

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from vhoi.data_loading import (
    segmentation_from_output_class, 
    compute_centroid, 
    ignore_last_step_end_flag_general, 
    smooth_segmentation, 
    maybe_scale_input_tensors, 
)
from pyrutils.torch.train_utils import numpy_to_torch

def create_data(feature_dirs, action_classes, new_action_classes=None, 
                downsampling: int = 1, test_data: bool = False):
    human_features_list = []
    human_boxes_list = []
    human_poses_list = []
    object_features_list = []
    object_boxes_list = []
    gt_list = [] if not test_data else None
    xs_steps = []

    for feature_dir in feature_dirs:
        feature_dir = Path(feature_dir)
        
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
        
        # Store number of steps
        num_steps = len(subject_visual_features[downsampling - 1::downsampling])
        xs_steps.append(num_steps)

    xs_steps = np.array(xs_steps, dtype=np.float32)

    return (
        human_features_list,
        human_boxes_list,
        human_poses_list,
        object_features_list,
        object_boxes_list,
        gt_list,
        xs_steps,
    )

def assemble_mphoi_frame_level_recurrent_human(
    human_features_list, human_poses_list, object_boxes_list, gt_list,
    downsampling: int = 1, 
    max_no_objects: int = 4
):
    xs_h, xs_hp, x_obb = [], [], []
    max_len, max_len_downsampled = 0, 0

    if max_no_objects is None:
        max_no_objects = max(
            max(len(frame) for frame in video) for video in object_boxes_list
        )

    for humans, poses, objects_bounding_box in zip(human_features_list, human_poses_list, object_boxes_list):
        num_humans = len(humans)
        max_len = max(max_len, humans[0].shape[0])

        humans_ds = [h[downsampling - 1::downsampling] for h in humans]
        poses_ds  = [p[downsampling - 1::downsampling] / 1000 for p in poses]
        max_len_downsampled = max(
            max_len_downsampled,
            max(h.shape[0] for h in humans_ds),
            max(p.shape[0] for p in poses_ds),
            objects_bounding_box[downsampling - 1::downsampling].shape[0],
        )
        xs_h.append(humans_ds)
        xs_hp.append(poses_ds)

        obb_ds = objects_bounding_box[downsampling - 1::downsampling] / 1000
        x_obb.append(obb_ds)

    xs_obb = []
    for video in x_obb:
        bb = []
        for frame in video:
            b = np.zeros((max_no_objects, 4))
            n = min(len(frame), max_no_objects)
            b[:n] = frame[:n]
            b = b.reshape(max_no_objects * 2, 2)
            bb.append(b)
        xs_obb.append(bb)
    
    keypoints = list(range(human_poses_list[0][0].shape[1]))
    xs_h_with_context = []
    for i, (humans_ds, poses_ds, obb_video) in enumerate(zip(xs_h, xs_hp, xs_obb)):
        num_humans = len(humans_ds)
        humans_context = [[] for _ in range(num_humans)]

        for j in range(len(humans_ds[0])):
            obb = obb_video[j]

            if j + 1 < len(humans_ds[0]):
                next_poses = [p[j+1][keypoints] for p in poses_ds]
                pose_velos = [(next_pose - poses_ds[h][j][keypoints]) * 100 for h, next_pose in enumerate(next_poses)]
                obb_velo = (obb_video[j+1] - obb) * 100
            else:
                pose_velos = [np.zeros((len(keypoints), 2)) for _ in poses_ds]
                obb_velo = np.zeros((max_no_objects * 2, 2))

            obbvelo = np.hstack((obb, obb_velo)).reshape(1, -1)

            context = []
            for h in range(num_humans):
                pose = poses_ds[h][j][keypoints]
                velo = pose_velos[h]
                posevelo = np.hstack((pose, velo)).reshape(1, -1)
                context.append(posevelo[0])

            context = np.concatenate(context + [obbvelo[0]])

            for h in range(num_humans):
                h_con = np.concatenate((humans_ds[h][j], context))
                humans_context[h].append(h_con)

        xs_h_with_context.append([np.array(hc) for hc in humans_context])

    feature_size = xs_h_with_context[0][0].shape[-1]
    num_humans = len(xs_h_with_context[0])
    x_hs = np.full([len(xs_h_with_context), max_len_downsampled, num_humans, feature_size],
                   fill_value=np.nan, dtype=np.float32)

    for m, humans in enumerate(xs_h_with_context):
        for h, feats in enumerate(humans):
            seq_len = min(len(feats), max_len_downsampled)
            x_hs[m, :seq_len, h] = feats[:seq_len]

    xs = [x_hs]

    # Outputs
    y_rec_hs = np.full([len(x_hs), max_len_downsampled, num_humans], fill_value=-1, dtype=np.int64)
    y_pred_hs = np.full_like(y_rec_hs, fill_value=-1)
    if gt_list:
        for m, video_hands_ground_truth in enumerate(gt_list):
            for h in range(num_humans):
                human_key = f"Human{h+1}"
                if human_key not in video_hands_ground_truth:
                    continue

                y_h = video_hands_ground_truth[human_key]

                # Ground truth (downsampled)
                y_h_ds = y_h[downsampling - 1::downsampling]
                seq_len = min(len(y_h_ds), y_rec_hs.shape[1])
                y_rec_hs[m, :seq_len, h] = y_h_ds[:seq_len]

                # Prediction: shift labels forward
                y_h_p = np.roll(y_h, -1)
                y_h_p[-1] = -1  # last frame has no "next"
                y_h_p_ds = y_h_p[downsampling - 1::downsampling]
                seq_len_p = min(len(y_h_p_ds), y_pred_hs.shape[1])
                y_pred_hs[m, :seq_len_p, h] = y_h_p_ds[:seq_len_p]
                # y_pred_hs[m, :seq_len, h] = y_h_ds[:seq_len]       
    x_hs_segmentation = segmentation_from_output_class(y_rec_hs, segmentation_type="input")
    xs.append(x_hs_segmentation)
    y_hs_segmentation = segmentation_from_output_class(y_rec_hs, segmentation_type="output")
    ys = [y_rec_hs, y_pred_hs, y_hs_segmentation]
    return xs, ys

def assemble_mphoi_frame_level_recurrent_objects(object_features_list, downsampling: int = 1):
    xs_objects = []
    max_len, max_len_downsampled, max_num_objects = 0, 0, 0
    for objects in object_features_list:
        max_len = max(max_len, objects.shape[0])
        max_num_objects = max(max_num_objects, objects.shape[1])
        objects = objects[downsampling - 1::downsampling]
        max_len_downsampled = max(max_len_downsampled, objects.shape[0])
        xs_objects.append(objects)
    feature_size = xs_objects[-1].shape[-1]
    x_objects = np.full([len(xs_objects), max_len_downsampled, max_num_objects, feature_size],
                        fill_value=np.nan, dtype=np.float32)
    x_objects_mask = np.zeros([len(xs_objects), max_num_objects], dtype=np.float32)
    for m, x_o in enumerate(xs_objects):
        x_objects[m, :x_o.shape[0], :x_o.shape[1], :] = x_o
        x_objects_mask[m, :x_o.shape[1]] = 1.0
    xs = [x_objects, x_objects_mask]
    return xs

def assemble_mphoi_human_human_distances(human_boxes_list, downsampling: int = 1):
    """
    Compute pairwise human-human distances for multiple humans across videos.

    Args:
        human_boxes_list: list of list of human bounding boxes per video
                          (outer list: videos, inner list: humans, array: frames x 4)
        downsampling:     frame downsampling factor

    Returns:
        x_hh_dists: tensor [num_videos, max_len, N, N] with pairwise distances
    """
    mphoi_dims = np.array([3840, 2160], dtype=np.float32)
    max_len, max_num_humans = 0, 0
    all_dists = []

    for video_bbs in human_boxes_list:
        num_humans = len(video_bbs)
        max_num_humans = max(max_num_humans, num_humans)

        # Downsample and compute centroids
        centroids = []
        for bb in video_bbs:
            bb = bb[downsampling - 1::downsampling]
            c = compute_centroid(bb) / mphoi_dims
            centroids.append(c)

        # Length of this video (frames)
        max_len = max(max_len, centroids[0].shape[0])

        # Compute pairwise distances (frames x N x N)
        T = centroids[0].shape[0]
        dists_matrix = np.zeros((T, num_humans, num_humans), dtype=np.float32)

        for i, j in itertools.combinations(range(num_humans), 2):
            d = np.linalg.norm(centroids[i] - centroids[j], ord=2, axis=-1)
            dists_matrix[:, i, j] = d
            dists_matrix[:, j, i] = d

        all_dists.append(dists_matrix)

    # Pad into a tensor [num_videos, max_len, max_num_humans, max_num_humans]
    tensor_shape = [len(all_dists), max_len, max_num_humans, max_num_humans]
    x_hh_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)

    for m, dists_matrix in enumerate(all_dists):
        T, N, _ = dists_matrix.shape
        x_hh_dists[m, :T, :N, :N] = dists_matrix

    return x_hh_dists

def assemble_mphoi_human_object_distances(human_boxes_list, object_boxes_list, downsampling: int = 1):
    """
    Compute human-object distances for multiple humans and objects across videos.

    Args:
        human_boxes_list:  list of list of human bounding boxes per video
                           (outer list: videos, inner list: humans, array: frames x 4)
        object_boxes_list: list of object bounding box arrays per video (frames x num_objects x 4)
        downsampling:      frame downsampling factor

    Returns:
        x_ho_dists: tensor [num_videos, max_len, max_num_humans, max_num_objects]
    """
    mphoi_dims = np.array([3840, 2160], dtype=np.float32)
    max_len, max_num_humans, max_num_objects = 0, 0, 0
    all_dists = []

    for video_bbs, obj_bbs in zip(human_boxes_list, object_boxes_list):
        num_humans = len(video_bbs)

        # Downsample humans → centroids
        human_centroids = []
        for bb in video_bbs:
            bb = bb[downsampling - 1::downsampling]
            c = compute_centroid(bb) / mphoi_dims
            human_centroids.append(c)

        # Downsample objects → centroids
        obj_bbs = obj_bbs[downsampling - 1::downsampling]
        obj_centroids = compute_centroid(obj_bbs) / mphoi_dims

        T = obj_centroids.shape[0]
        max_len = max(max_len, T)
        max_num_humans = max(max_num_humans, num_humans)
        max_num_objects = max(max_num_objects, obj_centroids.shape[1])

        # Compute distances [frames, num_humans, num_objects]
        dists_matrix = np.zeros((T, num_humans, obj_centroids.shape[1]), dtype=np.float32)
        for h, h_c in enumerate(human_centroids):
            d = np.linalg.norm(obj_centroids - np.expand_dims(h_c, axis=1), ord=2, axis=-1)
            dists_matrix[:, h, :] = d

        all_dists.append(dists_matrix)

    # Pad into a tensor [num_videos, max_len, max_num_humans, max_num_objects]
    tensor_shape = [len(all_dists), max_len, max_num_humans, max_num_objects]
    x_ho_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)

    for m, dists_matrix in enumerate(all_dists):
        T, H, O = dists_matrix.shape
        x_ho_dists[m, :T, :H, :O] = dists_matrix

    return x_ho_dists

def assemble_mphoi_object_object_distances(object_boxes_list, downsampling: int = 1):
    """
    Compute pairwise object-object distances across videos.

    Args:
        object_boxes_list: list of object bounding box arrays per video (frames x num_objects x 4)
        downsampling:      frame downsampling factor

    Returns:
        x_oo_dists: tensor [num_videos, max_len, max_num_objects, max_num_objects]
    """
    mphoi_dims = np.array([3840, 2160], dtype=np.float32)
    max_len, max_num_objects = 0, 0
    all_dists = []

    for obj_bbs in object_boxes_list:
        # Downsample and compute centroids
        obj_bbs = obj_bbs[downsampling - 1::downsampling]
        objs_centroid = compute_centroid(obj_bbs) / mphoi_dims # (frames, num_objects, 2)
        num_objects = objs_centroid.shape[1]

        # Compute pairwise distances per frame
        dists = []
        for k in range(num_objects):
            kth_object_centroid = objs_centroid[:, k:k+1]  # (frames, 1, 2)
            kth_dist = np.linalg.norm(objs_centroid - kth_object_centroid, ord=2, axis=-1) # (frames, num_objects)
            dists.append(kth_dist)

        dists = np.stack(dists, axis=1)  # (frames, num_objects, num_objects)
        all_dists.append(dists)

        max_len = max(max_len, obj_bbs.shape[0])
        max_num_objects = max(max_num_objects, num_objects)

    # Pad into tensor [num_videos, max_len, max_num_objects, max_num_objects]
    tensor_shape = [len(all_dists), max_len, max_num_objects, max_num_objects]
    x_oo_dists = np.full(tensor_shape, fill_value=np.nan, dtype=np.float32)

    for m, dists in enumerate(all_dists):
        T, O1, O2 = dists.shape
        x_oo_dists[m, :T, :O1, :O2] = dists

    return x_oo_dists

def assemble_mphoi_tensors(
    human_features_list,
    human_boxes_list,
    human_poses_list,
    object_features_list,
    object_boxes_list,
    gt_list,
    xs_steps,
    model_name: str, 
    sigma: float = 0.0, 
    downsampling: int = 1,
):
    xs, ys = assemble_mphoi_frame_level_recurrent_human(human_features_list, human_poses_list, object_boxes_list, gt_list)
    xs_objects = assemble_mphoi_frame_level_recurrent_objects(object_features_list, downsampling=downsampling)
    if model_name == '2G-GCN':
        if sigma:
            ys[2] = ignore_last_step_end_flag_general(ys[2])
        ys[2] = smooth_segmentation(ys[2], sigma)
        ys_budget = ys[2]
        xs_hh_dists = assemble_mphoi_human_human_distances(human_boxes_list, downsampling=downsampling)
        xs_ho_dists = assemble_mphoi_human_object_distances(human_boxes_list, object_boxes_list, downsampling=downsampling)
        xs_oo_dists = assemble_mphoi_object_object_distances(object_boxes_list, downsampling=downsampling)
        xs = xs[:1] + xs_objects + xs[1:] + [xs_hh_dists, xs_ho_dists, xs_oo_dists, xs_steps]
        ys = [ys_budget] + ys[2:] + ys[:2]
        ys += ys[-2:]
    else:
        raise ValueError(f'MPHOI code not implemented for {model_name} yet.')
    return xs, ys

def create_data_loader(
    human_features_list,
    human_boxes_list,
    human_poses_list,
    object_features_list,
    object_boxes_list,
    gt_list,
    xs_steps,
    model_name: str, 
    batch_size: int, 
    shuffle: bool,
    scaling_strategy: Optional[str] = None, 
    scalers: Optional[dict] = None, 
    sigma: float = 0.0,
    downsampling: int = 1, 
):
    x, y = assemble_mphoi_tensors(
        human_features_list,
        human_boxes_list,
        human_poses_list,
        object_features_list,
        object_boxes_list,
        gt_list,
        xs_steps,
        model_name=model_name, 
        sigma=sigma, 
        downsampling=downsampling, 
    )
    
    x, scalers = maybe_scale_input_tensors(x, model_name, scaling_strategy=scaling_strategy, scalers=scalers)
    x = [np.nan_to_num(ix, copy=False, nan=0.0) for ix in x]
    x, y = numpy_to_torch(*x), numpy_to_torch(*y)
    dataset = TensorDataset(*(x + y))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                             pin_memory=False, drop_last=False)
    segmentations = None
    return data_loader, scalers, segmentations
