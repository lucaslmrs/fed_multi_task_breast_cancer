import logging
import yaml
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

from src.dataset.BUSI_dataloader import load_datasets
from src.dataset.classic_dataloader import classic_data_config, classic_loss_config
from src.utils.criterions import apply_criterion_multitask_segmentation_classification
from src.utils.experiment_init import device_setup
from src.utils.experiment_init import load_multitask_experiment_artefacts
from src.utils.metrics import binary_classification_metrics
from src.utils.metrics import dice_score_from_tensor
from src.utils.metrics import multiclass_classification_metrics
from src.utils.miscellany import init_log
from src.utils.miscellany import load_config_file
from src.utils.miscellany import load_config
from src.utils.miscellany import save_classification_results
from src.utils.miscellany import save_segmentation_results
from src.utils.miscellany import seed_everything
from src.utils.miscellany import write_metrics_file
from src.utils.models import inference_multitask_binary_classification_segmentation
from src.utils.models import inference_multitask_multiclass_classification_segmentation
from src.utils.models import load_pretrained_model
from src.utils.visualization import plot_evolution
from src.utils.training_runtime import (
    RuntimeEvents,
    dataloader_kwargs,
    precision_policy,
    start_gpu_telemetry,
)


def processes_classification_predicted(num_classes, pred_logits, gt_label, gt_list, pred_list):
    # averaging prediction if deep supervision
    if isinstance(pred_logits, list):
        pred_logits = torch.mean(torch.stack(pred_logits, dim=0), dim=0)

    # this if-else differentiates between multiclass and binary class predictions
    if num_classes > 2:
        # applying softmax to get probabilities
        probabilities = torch.nn.functional.softmax(pred_logits, dim=1)

        # Applying argmax to get the class with the highest probability
        gt_label = [torch.argmax(k, keepdim=True).to(torch.float) for k in gt_label]
        pred_class = [torch.argmax(pl, keepdim=True).to(torch.float) for pl in probabilities]

        # storing the probabilities and ground truth labels in lists
        for la, p in zip(gt_label, pred_class):
            gt_list.append(la.detach().item())
            pred_list.append(p.detach().item())
    else:
        # adding ground truth label and predicted label
        if pred_logits.shape[0] > 1:  # when batch size > 1, each element is added individually
            for i in range(pred_logits.shape[0]):
                pred_list.append((torch.sigmoid(pred_logits[i, :]) > .5).double().detach().item())
                gt_list.append(gt_label[i, :].detach().item())
        else:
            pred_list.append((torch.sigmoid(pred_logits) > .5).double().detach().item())
            gt_list.append(gt_label.detach().item())

    return gt_list, pred_list


def process_segmentation_predicted(outputs, masks):
    # measuring DICE error
    if isinstance(outputs, list):
        outputs = outputs[-1]
    outputs = torch.sigmoid(outputs) > .5  # converting continuous values into probability [0, 1]

    return dice_score_from_tensor(masks, outputs)


def _epoch(num_classes, training):
    from src.utils.classic_multitask import run_epoch
    result = run_epoch(model, training_loader if training else validation_loader, dev, precision,
                       num_classes, seg_criterion, cls_criterion, alpha,
                       config_loss['inversely_weighted'], optimizer if training else None)
    epoch_metrics['train' if training else 'val'] = result
    values = (result['loss'], result['dice'], result['acc'], result['f1'])
    return values if training else values + (result['seg_loss'], result['cls_loss'])


epoch_metrics = {}


def train_one_epoch(num_classes):
    return _epoch(num_classes, True)


@torch.inference_mode()
def validate_one_epoch(num_classes):
    return _epoch(num_classes, False)


# alphas = [1, .95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, .4, .35, .3, .25, .2, .15, .1, .05, .0]
# # alphas = [.25, .2, .15, .1, .05, .0, -1, -2, -5]
# for alpha in alphas:
# for beta in betas:
# beta = 5
# print(beta)


def run(config_path="./src/config.yaml", run_path=None):
    global config_model, config_opt, config_loss, config_training, config_data, dev, alpha, training_loader, validation_loader, test_loader, model, optimizer, seg_criterion, cls_criterion, precision
    # initializing times
    init_time = time.perf_counter()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # loading config file
    full_config = load_config(config_path)
    config_model, config_opt, config_loss, config_training, config_data = load_config_file(path=config_path)
    config_data = classic_data_config(full_config)
    config_loss = classic_loss_config(full_config)
    config_model["sequences"] = config_data.get("channels", config_model["sequences"])
    if config_training['CV'] < 1:
        sys.exit("training.CV must be at least 1 (CV=1 selects deterministic holdout)")

    # initializing seed and gpu if possible
    seed_everything(config_training['seed'], cuda_benchmark=config_training['cuda_benchmark'])
    dev = device_setup()
    precision = precision_policy(full_config, dev)

    # initializing folder structures and log
    # config_training['alpha'] = alpha
    alpha = config_training['alpha']
    if run_path is None:
        run_path = f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}_alpha_{config_training['alpha']}" \
                   f"_batch_{config_data['batch_size']}_{'_'.join(config_data['classes'])}"
    resolved_run_path = Path(run_path)
    resolved_run_path.mkdir(parents=True, exist_ok=True)
    init_log(log_name=str(resolved_run_path / "execution.log"))
    full_config['data'], full_config['model'] = config_data, config_model
    (resolved_run_path / 'config.yaml').write_text(
        yaml.safe_dump(full_config, sort_keys=False), encoding='utf-8')
    run_path = str(resolved_run_path)
    gpu_telemetry = start_gpu_telemetry(full_config, run_path)
    runtime_events = RuntimeEvents(Path(run_path) / "runtime_events.csv", dev)

    # initializing experiment's objects
    n_classes = len(config_data['classes'])
    n_augments = sum([v for k, v in config_data['augmentation'].items()])
    transforms = torch.nn.Sequential(
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=360)
    )
    train_loaders, val_loaders, test_loaders = load_datasets(
        config_training, config_data, transforms, mode='CV', tasks=('seg', 'cls'),
        runtime={
            "loader_options": dataloader_kwargs(full_config),
            "inference_batch_size": full_config.get("runtime", {}).get(
                "inference_batch_size", 32
            ),
        },
    )


    for n, (training_loader, validation_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
        logging.info(f"\n\n *********************  FOLD {n}  ********************* \n\n")
        logging.info(f"\n\n ###############  TRAINING PHASE  ###############  \n\n")

        # creating specific paths and experiment's objects for each fold
        fold_time = time.perf_counter()
        Path(f"{run_path}/fold_{n}/segs/").mkdir(parents=True, exist_ok=True)
        Path(f"{run_path}/fold_{n}/plots/").mkdir(parents=True, exist_ok=True)
        Path(f"{run_path}/fold_{n}/features_map/").mkdir(parents=True, exist_ok=True)

        # artefacts initialization
        if config_data.get('class_weighting', 'balanced_fold') == 'balanced_fold':
            config_data['classes_weighted'] = None
            config_data['class_weights'] = training_loader.dataset.class_weights
        model, optimizer, seg_criterion, cls_criterion, scheduler = load_multitask_experiment_artefacts(config_data, config_model, config_opt, config_loss, n_augments, run_path, device=dev)
        model = model.to(dev)
        (Path(run_path) / f'fold_{n}/supervision_metadata.yaml').write_text(yaml.safe_dump({
            'class_names': config_data['classes'],
            'class_weights': config_data.get('class_weights'),
            'class_weight_source': 'supervised_training_rows_only',
            'oversampling': False,
        }), encoding='utf-8')

        # init metrics file
        write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                           text_to_write=f'epoch,LR,Train_loss,Validation_loss,Train_dice,Validation_dice,Train_acc,Train_F1,Validation_acc,Validation_F1')

        best_validation_loss = 1_000_000.
        patience = 0
        for epoch in range(config_training['epochs']):
            current_lr = optimizer.param_groups[0]["lr"]
            start_epoch_time = time.perf_counter()

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            with runtime_events.measure("train_epoch", setup="multitask", fold=n, epoch=epoch) as event:
                avg_train_loss, avg_dice, train_acc, train_f1_score = train_one_epoch(n_classes)
                event.update(examples=len(training_loader.dataset), batches=len(training_loader))

            # We don't need gradients on to do reporting
            model.train(False)
            with runtime_events.measure("validation_epoch", setup="multitask", fold=n, epoch=epoch) as event:
                avg_validation_loss, avg_validation_dice, val_acc_score, val_f1_score, segmentation_val_loss, classification_val_loss = validate_one_epoch(n_classes)
                event.update(examples=len(validation_loader.dataset), batches=len(validation_loader))

            # # Update the learning rate at the end of each epoch
            if config_opt['scheduler'] == 'cosine':
                scheduler.step()
            else:
                scheduler.step(avg_validation_loss)

            # Track the best performance, and save the model's state
            if avg_validation_loss < best_validation_loss:
                patience = 0  # restarting patience
                best_validation_loss = avg_validation_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': 'scheduler',
                    'val_loss': best_validation_loss
                }, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')
            else:
                patience += 1

            # logging results of current epoch
            end_epoch_time = time.perf_counter()
            logging.info(f'EPOCH {epoch} --> '
                         f'|| Training loss {avg_train_loss:.4f} '
                         f'|| Validation loss {avg_validation_loss:.4f} '
                         f'|| Segmentation val loss {segmentation_val_loss:.4f} '
                         f'|| Classification val loss {classification_val_loss:.4f} '
                         f'|| Training DICE {avg_dice:.4f} '
                         f'|| Validation DICE  {avg_validation_dice:.4f} '
                         f'|| Training ACC {train_acc:.4f} '
                         f'|| Training F1 {train_f1_score:.4f} '
                         f'|| Validation ACC {val_acc_score:.4f} '
                         f'|| Validation F1 {val_f1_score:.4f} '
                         f'|| Patience: {patience} '
                         f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f}'
                         f'|| Best validation performance: {best_validation_loss:.4f}'
            )

            write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                               text_to_write=f'{epoch},{current_lr:.8f},{avg_train_loss:.4f},{avg_validation_loss:.4f},'
                                             f'{avg_dice:.4f}, {avg_validation_dice:.4f},{train_acc:.4f},'
                                             f'{train_f1_score:.4f},{val_acc_score:.4f},{val_f1_score:.4f}',
                               close=True)

            pd.DataFrame([
                dict(epoch=epoch, split=split, **values) for split, values in epoch_metrics.items()
            ]).to_csv(Path(run_path) / f'fold_{n}/supervision_metrics.csv', mode='a',
                      header=epoch == 0, index=False)

            # early stopping
            if patience > config_training['max_patience']:
                logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
                break

        # store metrics
        metrics = pd.read_csv(f'{run_path}/fold_{n}/metrics.csv')
        plot_evolution(metrics, columns=['Train_loss', 'Validation_loss'], path=f'{run_path}/fold_{n}/loss_evolution.png')
        plot_evolution(metrics, columns=['Train_dice', 'Validation_dice'], path=f'{run_path}/fold_{n}/segmentation_metrics_evolution.png')
        plot_evolution(metrics, columns=['Train_acc', 'Train_F1', 'Validation_acc', 'Validation_F1'], path=f'{run_path}/fold_{n}/classification_metrics_evolution.png')

        """
        INFERENCE PHASE
        """

        # results for validation dataset
        logging.info(f"\n\n ###############  VALIDATION PHASE  ###############  \n\n")
        model = load_pretrained_model(model, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')

        # results for test dataset
        logging.info(f"\n\n ###############  TESTING PHASE  ###############  \n\n")
        with runtime_events.measure("test", setup="multitask", fold=n) as event:
            with precision.autocast():
                if not (config_training.get('overlap_seg_based_on_class', False) or
                        config_training.get('overlap_class_based_on_seg', False)):
                    from src.utils.classic_multitask import inference_supervised_multitask
                    test_results_segmentation, test_results_classification = inference_supervised_multitask(
                        model, test_loader, f'{run_path}/fold_{n}', dev, precision)
                elif len(config_data['classes']) <= 2:
                    test_results_segmentation, test_results_classification = inference_multitask_binary_classification_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev)
                else:
                    test_results_segmentation, test_results_classification = inference_multitask_multiclass_classification_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev, threshold=config_training["threshold_postprocessing"], overlap_seg_based_on_class=config_training["overlap_seg_based_on_class"], overlap_class_based_on_seg=config_training["overlap_class_based_on_seg"])
            event.update(examples=len(test_loader.dataset), batches=len(test_loader))
        logging.info(f"Segmentation metric:\n\n{test_results_segmentation.mean()}\n")

        # classification metrics
        if test_results_classification.empty:
            logging.info('No classification supervision in test slice')
        elif len(config_data['classes']) <= 2:
            logging.info(f"\nClassification metrics:\n\n{pformat(binary_classification_metrics(test_results_classification.ground_truth, test_results_classification.predicted_label))}")
        else:
            logging.info(f"\nClassification metrics:\n\n{pformat(multiclass_classification_metrics(test_results_classification.ground_truth, test_results_classification.predicted_label, labels=list(range(len(config_data['classes'])))))}")

        # Clear the GPU memory after evaluating on the test data for this fold
        torch.cuda.empty_cache()

        del model


    # saving final results as an Excel file
    save_segmentation_results(run_path, n_splits=config_training['CV'])

    # saving final results as an Excel file
    save_classification_results(run_path, len(config_data['classes']), n_splits=config_training['CV'])

    # Measuring total time
    end_time = time.perf_counter()
    logging.info(f"Total time for all of the folds: {end_time - init_time:.2f}")
    runtime_events.write()
    gpu_telemetry.stop()


if __name__ == "__main__":
    run()
