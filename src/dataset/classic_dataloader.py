"""Single-dataset loaders with sample-level supervision and group-safe splits."""
import copy

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import paths
from src.dataset.BUSI_dataset import BUSI
from src.dataset.federated_partition import carve_validation
from src.dataset.splitting import outer_split_indices


def present(series):
    return series.notna() & series.astype(str).str.strip().ne('')


def classic_data_config(config):
    data = copy.deepcopy(config['data'])
    registry = config.get('datasets', {}).get(data['dataset'], {})
    for key in ('channels', 'cls_source_split', 'class_weighting'):
        if key in registry:
            data[key] = registry[key]
    # Retain explicit class subsets; replacing data.dataset alone must not retain BUSI labels.
    if registry.get('classes') and not set(data.get('classes', [])).issubset(registry['classes']):
        data['classes'] = list(registry['classes'])
    return data


def classic_loss_config(config):
    """Global ``loss`` with the active dataset's classification overrides, as the federated client."""
    loss = copy.deepcopy(config['loss'])
    registry = config.get('datasets', {}).get(config['data']['dataset'], {})
    for key in ('classification_criterion', 'focal_gamma'):
        if key in registry:
            loss[key] = registry[key]
    return loss


def supervision_frame(frame, data, tasks):
    frame = frame.copy()
    for col in ('mask_path', 'class'):
        if col not in frame:
            frame[col] = None
    mask, label = present(frame.mask_path), present(frame['class'])
    if (~(mask | label)).any():
        raise ValueError('Images without mask and class are not supervised training examples')
    if frame.img_path.duplicated().any():
        raise ValueError('Classic mapping must contain one row per image')
    # Filter configured diagnosis classes without discarding segmentation-only images.
    frame = frame[~label | frame['class'].isin(data['classes'])].copy()
    if data.get('cls_source_split') and 'official_split' in frame:
        frame = frame[~present(frame['class']) | frame.official_split.eq(data['cls_source_split'])].copy()
    frame.loc[frame['class'].isin(data.get('seg_exclude_classes', [])), 'mask_path'] = None
    mask, label = present(frame.mask_path), present(frame['class'])
    selected = (mask if 'seg' in tasks else False) | (label if 'cls' in tasks else False)
    return frame[selected].reset_index(drop=True)


def split_supervised_frame(frame, training, data):
    """Split once for labeled images (including jointly annotated ones), once for mask-only.

    Known groups may never straddle these pools. Missing group IDs are distinct image groups.
    The ungrouped fully labeled path retains the historical BUSI split and validation seeds.
    """
    frame = frame.copy()
    frame['_pool'] = np.where(present(frame['class']), 'labeled', 'mask_only')
    grouped = 'lesion_id' in frame and present(frame.lesion_id).any()
    if grouped:
        known = frame[present(frame.lesion_id)]
        if (known.groupby('lesion_id')['_pool'].nunique() > 1).any():
            raise ValueError('A lesion group crosses independently split supervision pools')
        frame['_group'] = [f'lesion:{g}' if pd.notna(g) and str(g).strip() else f'image:{p}'
                           for g, p in zip(frame.lesion_id, frame.img_path)]
    folds = [[] for _ in range(training['CV'])]
    for kind, pool in frame.groupby('_pool', sort=False):
        strategy = 'stratified' if kind == 'labeled' else 'random'
        group_col = '_group' if grouped else None
        split_pool = pool
        if grouped:
            # Mask-only groups have no diagnosis; stratification over one constant label is safe.
            split_pool = pool.copy()
            if kind == 'mask_only':
                split_pool['class'] = '__mask_only__'
            strategy = 'stratified_group'
        outer = outer_split_indices(split_pool, n_splits=training['CV'], seed=training['seed'],
                                    strategy=strategy, group_col=group_col,
                                    holdout_test_size=training.get('holdout_test_size', .30))
        for fold, (train_ix, test_ix) in enumerate(outer):
            train, val = carve_validation(split_pool.iloc[train_ix], 1-data['train_size'],
                                          training['seed'], group_col=group_col)
            test = split_pool.iloc[test_ix].copy()
            if kind == 'mask_only':
                train, val = train.copy(), val.copy()
                for part in (train, val, test):
                    part['class'] = None
            folds[fold].append((train, val, test))
    result = []
    for pools in folds:
        parts = [pd.concat([p[i] for p in pools], ignore_index=True) for i in range(3)]
        for i in range(3):
            for j in range(i):
                for key in ('img_path', '_group') if grouped else ('img_path',):
                    if set(parts[i][key]) & set(parts[j][key]):
                        raise ValueError(f'{key} leaked between data splits')
        result.append(parts)
    return result


def class_weights_from_training(frame, classes):
    counts = frame.loc[present(frame['class']), 'class'].value_counts()
    total = counts.sum()
    return [float(total / (len(classes) * counts[c])) if counts.get(c, 0) else 0.0 for c in classes]


def load_classic_datasets(training, data, transforms, tasks, runtime=None):
    from src.dataset.BUSI_dataloader import deterministic_oversampling
    tasks = tuple(tasks)
    if len(tasks) > 1 and data.get('oversampling', False):
        raise ValueError('Multitask oversampling duplicates segmentation targets; use class weights')
    frame = supervision_frame(pd.read_csv(paths.require_mapping_file(data)), data, tasks)
    if frame.empty:
        raise ValueError('No supervised images for the requested tasks')
    runtime = runtime or {}
    loaders = [[], [], []]
    for fold, parts in enumerate(split_supervised_frame(frame, training, data)):
        for task, column in (('seg', 'mask_path'), ('cls', 'class')):
            if task in tasks and not present(parts[0][column]).any():
                raise ValueError(f'No training supervision for requested task {task}')
        weights = class_weights_from_training(parts[0], data['classes'])
        if data.get('oversampling', False):
            if not present(parts[0]['class']).all():
                raise ValueError('Class oversampling requires labels on every selected training image')
            parts[0] = deterministic_oversampling(parts[0])
        for index, part in enumerate(parts):
            part = part.copy()
            part['fold'] = fold
            dataset = BUSI(part, transforms=transforms if index == 0 else None,
                           augmentations=data.get('augmentation'), channels=data.get('channels', 1),
                           classes=data['classes'], dataset=data['dataset'])
            dataset.class_weights = weights
            batch_size = data['batch_size'] if index < 2 else runtime.get('inference_batch_size', 1)
            loaders[index].append(DataLoader(dataset, batch_size=batch_size, shuffle=index == 0,
                                             **runtime.get('loader_options', {})))
    return tuple(loaders)
