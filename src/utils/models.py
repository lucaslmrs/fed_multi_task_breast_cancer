import warnings
warnings.filterwarnings('ignore')
import numpy as np
import logging
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
from src.utils.metrics import calculate_metrics
from src.utils.metrics import calculate_metrics_multiclass_segmentation
from src.utils.images import count_pixels
from src.utils.images import postprocess_semantic_segmentation
from src.utils.images import postprocess_binary_segmentation
from scipy.ndimage import binary_fill_holes
from src.utils.training_runtime import move_to_device


def _batch_values(value):
    """Return collated metadata as a Python list without assuming batch size one."""
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_pretrained_model(model: nn.Module, ckpt_path: str):
    """
    It restores a pretrained state model

    :param model: PyTorch module to be used
    :param ckpt_path: Path to the checkpoint

    :return: Model with a state loaded
    """
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint '{ckpt_path}'. Last epoch: {checkpoint['epoch']}")
    else:
        raise ValueError(f"\n\t-> No checkpoint found at '{ckpt_path}'")

    return model


def inference_binary_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
        fill_holes: bool = True
):
    """
    It performs binary inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU
    :param fill_holes: fill holes in the predicted segmentation

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        labels = _batch_values(test_data['class'])
        test_images = move_to_device(test_data['image'], device)
        test_masks = move_to_device(test_data['mask'], device)

        # generating segmentation
        features_map = model(test_images)
        final_map = features_map[-1] if isinstance(features_map, list) else features_map
        test_outputs = (torch.sigmoid(final_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.float().detach().cpu().numpy()
        test_outputs = test_outputs.float().detach().cpu().numpy()

        for index, (patient_id, label) in enumerate(zip(patient_ids, labels)):
            maps = features_map if isinstance(features_map, list) else [features_map]
            for n, ds in enumerate(reversed(maps)):
                save_features_map(
                    seg=ds[index:index + 1],
                    path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png",
                )
            sample_mask = test_masks[index, 0]
            sample_output = test_outputs[index, 0]
            if fill_holes:
                sample_mask = sample_mask.astype(np.uint8)
                sample_output = binary_fill_holes(sample_output.astype(np.uint8)).astype(int)
            metrics = calculate_metrics(sample_mask, sample_output, patient_id)
            metrics['class'] = label
            results = results.append(metrics, ignore_index=True)
            save_binary_segmentation(
                seg=sample_output, path=f"{path}/segs/{label}_{patient_id}_seg.png"
            )

    # saving metrics results
    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    return results


def inference_multilabel_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
        postprocessing: bool = False
):
    """
    It performs multilabel inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled from 0 to the number of classes.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU
    :param postprocessing: boolean to decide whether labelling all the pixels as the majority class

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class', 'predicted_class'])

    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        labels = _batch_values(test_data['class'])
        test_images = move_to_device(test_data['image'], device)
        test_masks = move_to_device(test_data['mask'], device)

        # generating segmentation
        features_map = model(test_images)
        final_map = features_map[-1] if isinstance(features_map, list) else features_map
        test_outputs = torch.nn.functional.softmax(final_map, dim=1)

        # converting tensors to numpy arrays
        test_masks = torch.argmax(test_masks, dim=1, keepdim=True).float().float().detach().cpu().numpy()
        test_outputs = torch.argmax(test_outputs, dim=1, keepdim=True).float().float().detach().cpu().numpy()
        for index, (patient_id, label) in enumerate(zip(patient_ids, labels)):
            maps = features_map if isinstance(features_map, list) else [features_map]
            for n, ds in enumerate(reversed(maps)):
                save_features_map(
                    seg=ds[index:index + 1],
                    path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png",
                )
            sample_mask = test_masks[index:index + 1]
            sample_output = test_outputs[index:index + 1]
            processed = (
                postprocess_semantic_segmentation(sample_output)
                if postprocessing else sample_output
            )
            counter = count_pixels(sample_output)
            predicted_class = (
                'benign' if counter.get(1, 0) >= counter.get(2, 0) else 'malignant'
            )
            metrics = calculate_metrics_multiclass_segmentation(
                sample_mask, processed, patient_id
            )
            metrics['class'] = label
            metrics['predicted_class'] = predicted_class
            results = results.append(metrics, ignore_index=True)
            save_multilabel_segmentation(
                seg=sample_output, path=f"{path}/segs/{label}_{patient_id}_seg.png"
            )
            if postprocessing:
                save_multilabel_segmentation(
                    seg=processed,
                    path=f"{path}/segs/{label}_{patient_id}_seg_postprocessed.png",
                )

    # applying mapping for classification
    mapping_class = {
        'benign': 0,
        'malignant': 1
    }
    results['numerical_class'] = results['class'].map(mapping_class)
    results['numerical_class_predicted'] = results['predicted_class'].map(mapping_class)

    results.to_csv(f'{path}/results.csv', index=False)

    return results


def inference_multitask_binary_classification_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu'
):
    """
    It performs multitask inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1 as well as the image will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        labels = _batch_values(test_data['class'])
        test_images = move_to_device(test_data['image'], device)
        test_masks = move_to_device(test_data['mask'], device)

        # generating segmentation
        pred_class, features_map = model(test_images)
        final_map = features_map[-1] if isinstance(features_map, list) else features_map
        test_outputs = (torch.sigmoid(final_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.float().detach().cpu().numpy()
        test_outputs = test_outputs.float().detach().cpu().numpy()

        for index, (patient_id, label) in enumerate(zip(patient_ids, labels)):
            maps = features_map if isinstance(features_map, list) else [features_map]
            for n, ds in enumerate(reversed(maps)):
                save_features_map(
                    seg=ds[index:index + 1],
                    path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png",
                )
            sample_mask = test_masks[index:index + 1]
            sample_output = test_outputs[index:index + 1]
            metrics = calculate_metrics(sample_mask, sample_output, patient_id)
            metrics['class'] = label
            results = results.append(metrics, ignore_index=True)
            save_binary_segmentation(
                seg=sample_output, path=f"{path}/segs/{label}_{patient_id}_seg.png"
            )

    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    # classification
    patients = []
    ground_truth_label = []
    predicted_label = []
    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        labels = test_data['label'].float().detach().cpu().reshape(-1).tolist()
        test_images = move_to_device(test_data['image'], device)

        # generating segmentation
        test_outputs, segs = model(test_images)
        if isinstance(test_outputs, list):
            test_outputs = torch.mean(torch.stack(test_outputs, dim=0), dim=0)
        test_outputs = (torch.sigmoid(test_outputs) > .5).double()

        patients.extend(patient_ids)
        ground_truth_label.extend(labels)
        predicted_label.extend(test_outputs.float().detach().cpu().reshape(-1).tolist())

    # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return results, metrics


def inference_multitask_multiclass_classification_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
        threshold: int = 0,
        overlap_seg_based_on_class: bool = False,
        overlap_class_based_on_seg: bool = False
):
    """
    It performs multitask inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1 as well as the image will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU
    :param threshold: CPU or GPU
    :param overlap_seg_based_on_class: CPU or GPU
    :param overlap_class_based_on_seg: CPU or GPU

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        labels = _batch_values(test_data['class'])
        test_images = move_to_device(test_data['image'], device)
        test_masks = move_to_device(test_data['mask'], device)

        # generating segmentation
        pred_class, features_map = model(test_images)
        final_map = features_map[-1] if isinstance(features_map, list) else features_map
        logits = torch.mean(torch.stack(pred_class), dim=0) if isinstance(pred_class, list) else pred_class
        test_outputs = (torch.sigmoid(final_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.float().detach().cpu().numpy()
        test_outputs = test_outputs.float().detach().cpu().numpy()

        predicted_classes = logits.argmax(dim=1)
        for index, (patient_id, label) in enumerate(zip(patient_ids, labels)):
            maps = features_map if isinstance(features_map, list) else [features_map]
            for n, ds in enumerate(reversed(maps)):
                save_features_map(
                    seg=ds[index:index + 1],
                    path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png",
                )
            sample_mask = test_masks[index:index + 1]
            sample_output = test_outputs[index:index + 1]
            if threshold > 0:
                sample_output = postprocess_binary_segmentation(sample_output, threshold)
            if overlap_seg_based_on_class and predicted_classes[index].item() == 2:
                sample_output[sample_output > 0] = 0
            metrics = calculate_metrics(sample_mask, sample_output, patient_id)
            metrics['class'] = label
            results = results.append(metrics, ignore_index=True)
            save_binary_segmentation(
                seg=sample_output, path=f"{path}/segs/{label}_{patient_id}_seg.png"
            )

    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    # classification
    patients = []
    ground_truth_label = []
    predicted_label = []
    predicted_probabilities = []
    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        test_label = move_to_device(test_data['label'], device)
        test_label = torch.nn.functional.one_hot(test_label.flatten().to(torch.int64), num_classes=3).to(torch.float)
        test_images = move_to_device(test_data['image'], device)

        # generating segmentation
        test_outputs, segs = model(test_images)
        if isinstance(test_outputs, list):
            test_outputs = torch.mean(torch.stack(test_outputs, dim=0), dim=0)
        labels = test_label.argmax(dim=1).detach().cpu().tolist()
        probabilities = test_outputs.float().detach().cpu().tolist()
        predictions = test_outputs.argmax(dim=1).detach().cpu().tolist()
        final_segs = segs[-1] if isinstance(segs, list) else segs
        final_segs = (torch.sigmoid(final_segs) > .5).float().detach().cpu().numpy()
        for index, patient_id in enumerate(patient_ids):
            predicted = predictions[index]
            if overlap_class_based_on_seg and count_pixels(final_segs[index]).get(1, 0) == 0:
                predicted = 2
            patients.append(patient_id)
            ground_truth_label.append(int(labels[index]))
            predicted_label.append(int(predicted))
            predicted_probabilities.append(probabilities[index])

    # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })
    metrics[["prob_benign", "prob_malignant", "prob_normal"]] = predicted_probabilities
    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return results, metrics


def inference_multiclass_classification(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
):
    """
    It performs multitask inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1 as well as the image will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    # classification
    patients = []
    ground_truth_label = []
    predicted_label = []
    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        test_label = move_to_device(test_data['label'], device)
        test_label = torch.nn.functional.one_hot(test_label.flatten().to(torch.int64), num_classes=3).to(torch.float)
        test_images = move_to_device(test_data['image'], device)

        # generating segmentation
        test_outputs = model(test_images)
        if isinstance(test_outputs, list):
            test_outputs = torch.mean(torch.stack(test_outputs, dim=0), dim=0)
        patients.extend(patient_ids)
        ground_truth_label.extend(test_label.argmax(dim=1).detach().cpu().tolist())
        predicted_label.extend(test_outputs.argmax(dim=1).detach().cpu().tolist())

    # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return metrics


def inference_binary_classification(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu'
):
    """
    It performs binary classification inference over PyTorch dataloader by means of a trained model. It means the image
    will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    patients = []
    ground_truth_label = []
    predicted_label = []
    for test_data in test_loader:
        patient_ids = _batch_values(test_data['patient_id'])
        labels = test_data['label'].float().detach().cpu().reshape(-1).tolist()
        test_images = move_to_device(test_data['image'], device)

        # generating segmentation
        test_outputs = model(test_images)
        test_outputs = (torch.sigmoid(test_outputs) > .5).double()

        patients.extend(patient_ids)
        ground_truth_label.extend(labels)
        predicted_label.extend(test_outputs.float().detach().cpu().reshape(-1).tolist())

        # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results.csv', index=False)

    return metrics


def save_binary_segmentation(seg: np.array, path: str, value_non_zero: int = 255):
    """
    It saves a NumPy array as a binary image

    :param seg: Image to be saved
    :param path: path to save the image
    :param value_non_zero: value to assign all non-zero values. Typically, it will be 255 or 1.
    """

    n_dims = len(seg.shape)

    assert n_dims <= 4, "Numpy array must have less than 5 dimensions to be able to be stored"

    if n_dims == 4:
        seg = seg[0, 0, :, :].astype(int)
    elif n_dims == 3:
        seg = seg[0, :, :].astype(int)

    seg[seg > 0] = value_non_zero
    cv2.imwrite(path, seg)


def save_multilabel_segmentation(seg: np.array, path: str):
    """
    It saves a NumPy array as multilabel image

    :param seg: Image to be saved
    :param path: path to save the image
    """

    n_dims = len(seg.shape)

    assert n_dims <= 4, "Numpy array must have less than 5 dimensions to be able to be stored"

    if n_dims == 4:
        seg = seg[0, 0, :, :].astype(int)
    elif n_dims == 3:
        seg = seg[0, :, :].astype(int)

    if len(seg.shape) == 2:
        seg = seg.astype(int)
        cv2.imwrite(path, seg)
    else:
        seg = seg[0, 0, :, :].astype(int)
        cv2.imwrite(path, seg)


def save_features_map(seg: np.array, path: str):
    seg = seg.float().detach().cpu().numpy()
    seg = seg[0, 0, :, :].astype(float)
    cv2.imwrite(path, seg)


def count_parameters(model: torch.nn.Module) -> int:
    """
    This function counts the trainable parameters of a model.

    :param model: Torch model
    :return: Number of parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
