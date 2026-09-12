"""Classic multitask epochs and inference with explicit partial supervision."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.utils.metrics import calculate_metrics, summarize_segmentation_strata
from src.utils.supervision import _combined_loss, _supervised, _avg_logits


def run_epoch(model, loader, device, precision, num_classes, seg_criterion, cls_criterion,
              alpha, inversely_weighted, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, task_sums, task_counts = [], {'seg': 0., 'cls': 0.}, {'seg': 0, 'cls': 0}
    seg_rows, gt, pred = [], [], []
    with torch.set_grad_enabled(training):
        for data in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                logits, outputs = model(precision.move(data['image']))
                loss, counted, raw = _combined_loss(
                    data, logits, outputs, ('seg', 'cls'), {'seg': alpha, 'cls': 1-alpha},
                    device, num_classes, seg_criterion, cls_criterion, inversely_weighted)
            precision.ensure_finite(loss, 'classic multitask epoch')
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
            for task in counted:
                count = int(_supervised(data, task, device, len(data['image'])).sum())
                task_sums[task] += float(raw[task].detach()) * count
                task_counts[task] += count
            seg = outputs[-1] if isinstance(outputs, list) else outputs
            predicted = (seg.detach().float().sigmoid() > .5).cpu().numpy()
            masks = data['mask'].cpu().numpy()
            for i in _supervised(data, 'seg', 'cpu', len(masks)).nonzero().flatten().tolist():
                seg_rows.append(calculate_metrics(masks[i:i+1], predicted[i:i+1], str(i)))
            keep = _supervised(data, 'cls', device, len(masks))
            logits = _avg_logits(logits).detach()[keep]
            gt.extend(data['label'][keep.cpu()].flatten().long().tolist())
            pred.extend((logits.argmax(1) if num_classes > 2 else
                         (logits.sigmoid() > .5).flatten().long()).cpu().tolist())
    if not losses:
        raise ValueError('An epoch requires supervised batches')
    if training and any(count == 0 for count in task_counts.values()):
        raise ValueError(f'Requested task has no training supervision: {task_counts}')
    result = summarize_segmentation_strata(seg_rows)
    result.update(loss=float(np.mean(losses)),
                  dice=float(np.mean([r['DICE'] for r in seg_rows])) if seg_rows else np.nan,
                  acc=accuracy_score(gt, pred) if gt else np.nan,
                  f1=f1_score(gt, pred, labels=list(range(num_classes)), average='weighted', zero_division=0) if gt else np.nan)
    for task in ('seg', 'cls'):
        result[f'{task}_loss'] = task_sums[task]/task_counts[task] if task_counts[task] else np.nan
        result[f'{task}_samples'] = task_counts[task]
    return result


@torch.inference_mode()
def inference_supervised_multitask(model, loader, path, device, precision):
    from src.utils.models import _batch_values, save_binary_segmentation
    model.eval()
    root = Path(path)
    (root/'segs').mkdir(parents=True, exist_ok=True)
    seg_rows, cls_rows = [], []
    classes = loader.dataset.classes
    for data in loader:
        with precision.autocast():
            logits, outputs = model(precision.move(data['image']))
        logits = _avg_logits(logits).float()
        if logits.shape[1] == 1:
            positive = logits.sigmoid()
            probabilities = torch.cat([1-positive, positive], dim=1)
        else:
            probabilities = logits.softmax(1)
        probabilities = probabilities.cpu().numpy()
        seg = outputs[-1] if isinstance(outputs, list) else outputs
        predicted = (seg.float().sigmoid() > .5).cpu().numpy()
        masks = data['mask'].cpu().numpy()
        ids = _batch_values(data['patient_id'])
        has_mask = _supervised(data, 'seg', 'cpu', len(ids))
        has_label = _supervised(data, 'cls', 'cpu', len(ids))
        for i, patient in enumerate(ids):
            if has_mask[i]:
                row = calculate_metrics(masks[i:i+1], predicted[i:i+1], patient)
                row['class'] = data['class'][i]
                seg_rows.append(row)
                save_binary_segmentation(predicted[i:i+1], str(root/'segs'/f'{patient}_seg.png'))
            if has_label[i]:
                row = dict(patient_id=patient, ground_truth=int(data['label'][i].item()),
                           predicted_label=int(probabilities[i].argmax()))
                row.update({f'prob_{name}': float(p) for name, p in zip(classes, probabilities[i])})
                cls_rows.append(row)
    seg_frame = pd.DataFrame(seg_rows) if seg_rows else pd.DataFrame(columns=[
        'patient_id', 'class', 'DICE', 'Jaccard index', 'dice_positive', 'iou_positive',
        'empty_fp_image_rate', 'empty_predicted_area_fraction', 'n_positive', 'n_empty'])
    cls_frame = pd.DataFrame(cls_rows, columns=['patient_id', 'ground_truth', 'predicted_label']+
                             [f'prob_{name}' for name in classes])
    seg_frame.to_csv(root/'results_segmentation.csv', index=False)
    cls_frame.to_csv(root/'results_classification.csv', index=False)
    pd.DataFrame([summarize_segmentation_strata(seg_rows)]).to_csv(root/'segmentation_summary.csv', index=False)
    return seg_frame, cls_frame
