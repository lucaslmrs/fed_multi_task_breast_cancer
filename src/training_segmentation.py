import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms.v2 import RandomResizedCrop, ElasticTransform

from src.dataset.BUSI_dataloader import load_datasets
from src.utils.metrics import dice_score_from_tensor
from src.utils.miscellany import init_log
from src.utils.miscellany import seed_everything
from src.utils.models import inference_binary_segmentation
from src.utils.models import load_pretrained_model
from src.utils.visualization import plot_evolution
from src.utils.miscellany import save_segmentation_results
from src.utils.miscellany import load_config_file
from src.utils.miscellany import load_config
from src.utils.miscellany import write_metrics_file
from src.utils.criterions import apply_criterion_binary_segmentation
from src.utils.experiment_init import load_segmentation_experiment_artefacts
from src.utils.experiment_init import device_setup
from monai.transforms import MaskIntensity, HistogramNormalize, ThresholdIntensity
from src.utils.training_runtime import RuntimeEvents, dataloader_kwargs, precision_policy, start_gpu_telemetry

def train_one_epoch():
    training_loss = 0.
    running_dice = 0.

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, masks = precision.move(data['image']), precision.move(data['mask'])

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        with precision.autocast():
            outputs = model(inputs)
            batch_loss = apply_criterion_binary_segmentation(
                criterion=criterion, ground_truth=masks, segmentation=outputs,
                inversely_weighted=config_loss['inversely_weighted']
            )
        precision.ensure_finite(batch_loss, "classic segmentation training")
        training_loss += batch_loss.item()

        # Performing backward step through scaler methodology
        batch_loss.backward()
        optimizer.step()

        # measuring DICE
        if isinstance(outputs, list):
            outputs = outputs[-1]
        outputs = torch.sigmoid(outputs) > .5
        dice = dice_score_from_tensor(masks, outputs)
        running_dice += dice

        del batch_loss
        del outputs

    return training_loss / training_loader.__len__(), running_dice / training_loader.__len__()


@torch.no_grad()
def validate_one_epoch():
    validation_loss = 0.0
    validation_dice = 0.0
    for i, validation_data in enumerate(validation_loader):

        validation_images = precision.move(validation_data['image'])
        validation_masks = precision.move(validation_data['mask'])
        with precision.autocast():
            validation_outputs = model(validation_images)
            batch_validation_loss = apply_criterion_binary_segmentation(
                criterion=criterion, ground_truth=validation_masks,
                segmentation=validation_outputs,
                inversely_weighted=config_loss['inversely_weighted']
            )
        precision.ensure_finite(batch_validation_loss, "classic segmentation validation")
        validation_loss += batch_validation_loss.item()

        # measuring DICE
        if isinstance(validation_outputs, list):
            validation_outputs = validation_outputs[-1]
        validation_outputs = torch.sigmoid(validation_outputs) > .5
        dice = dice_score_from_tensor(validation_masks, validation_outputs)
        validation_dice += dice

    return validation_loss / validation_loader.__len__(), validation_dice / validation_loader.__len__()




def run(config_path="./src/config.yaml", run_path=None):
    global config_model, config_opt, config_loss, config_training, config_data, dev, training_loader, validation_loader, test_loader, model, optimizer, criterion, precision
    # initializing times
    init_time = time.perf_counter()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # loading config file
    full_config = load_config(config_path)
    config_model, config_opt, config_loss, config_training, config_data = load_config_file(path=config_path)
    if config_training['CV'] < 1:
        sys.exit("training.CV must be at least 1 (CV=1 selects deterministic holdout)")

    # initializing seed and gpu if possible
    seed_everything(config_training['seed'], cuda_benchmark=config_training['cuda_benchmark'])
    dev = device_setup()
    precision = precision_policy(full_config, dev)

    # initializing folder structures and log
    if run_path is None:
        run_path = (f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}_batch_"
                    f"{config_data['batch_size']}_{'_'.join(config_data['classes'])}")
    resolved_run_path = Path(run_path)
    resolved_run_path.mkdir(parents=True, exist_ok=True)
    init_log(log_name=str(resolved_run_path / "execution.log"))
    shutil.copyfile(config_path, resolved_run_path / "config.yaml")
    run_path = str(resolved_run_path)
    gpu_telemetry = start_gpu_telemetry(full_config, run_path)
    runtime_events = RuntimeEvents(Path(run_path) / "runtime_events.csv", dev)

    # initializing experiment's objects
    n_augments = sum([v for k, v in config_data['augmentation'].items()])
    training_transforms = torch.nn.Sequential(
        # transforms.RandomCrop(128, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # RandomResizedCrop(size=(128, 128)),
        transforms.RandomRotation(degrees=np.random.choice(range(0, 360))),
        # transforms.RandomCrop(64)
    )
    train_loaders, val_loaders, test_loaders = load_datasets(
        config_training, config_data, training_transforms, mode='CV',
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
        Path(f"{run_path}/fold_{n}/features_map/").mkdir(parents=True, exist_ok=True)
        Path(f"{run_path}/fold_{n}/plots/").mkdir(parents=True, exist_ok=True)

        # artefacts initialization
        model, optimizer, criterion, scheduler = load_segmentation_experiment_artefacts(config_model, config_opt,
                                                                                        config_loss, n_augments, run_path)
        model = model.to(dev)

        # init metrics file
        write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                           text_to_write=f'epoch,LR,Train,Validation,Train_loss,Val_loss')

        best_validation_loss = 1_000_000.
        patience = 0
        for epoch in range(config_training['epochs']):
            start_epoch_time = time.perf_counter()

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            with runtime_events.measure("train_epoch", setup="segmentation", fold=n, epoch=epoch) as event:
                avg_train_loss, avg_dice = train_one_epoch()
                event.update(examples=len(training_loader.dataset), batches=len(training_loader))

            # We don't need gradients on to do reporting
            model.train(False)
            with torch.no_grad():
                with runtime_events.measure("validation_epoch", setup="segmentation", fold=n, epoch=epoch) as event:
                    avg_validation_loss, avg_validation_dice = validate_one_epoch()
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
                }, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}.tar')
            else:
                patience += 1

            # logging results of current epoch
            end_epoch_time = time.perf_counter()
            logging.info(f'EPOCH {epoch} --> '
                         f'|| Training loss {avg_train_loss:.4f} '
                         f'|| Validation loss {avg_validation_loss:.4f} '
                         f'|| Training DICE {avg_dice:.4f} '
                         f'|| Validation DICE  {avg_validation_dice:.4f} '
                         # f'|| Test DICE  {results["DICE"].mean():.4f} '
                         f'|| Patience: {patience} '
                         f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f} '
                         f'|| LR: {optimizer.param_groups[0]["lr"]:.8f}')

            # write metrics
            write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                               text_to_write=f'{epoch},{optimizer.param_groups[0]["lr"]:.8f},'
                                             f'{avg_dice:.4f}, {avg_validation_dice:.4f},'
                                             f'{avg_train_loss:.4f},{avg_validation_loss:.4f}',
                               close=True)

            # early stopping
            if patience > config_training['max_patience']:
                logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
                break

        # store metrics
        metrics = pd.read_csv(f'{run_path}/fold_{n}/metrics.csv')
        plot_evolution(metrics, columns=['Train', 'Validation'],
                       path=f'{run_path}/fold_{n}/plots/metrics_evolution.png',
                       title='DICE coefficient', ylabel='DICE',)
        plot_evolution(metrics, columns=['Train_loss', 'Val_loss'],
                       path=f'{run_path}/fold_{n}/plots/loss_evolution.png',
                       title='DICE loss function', ylabel='Loss DICE',)

        """
        INFERENCE PHASE
        """

        # results for validation dataset
        # logging.info(f"\n\n ###############  VALIDATION PHASE  ###############  \n\n")
        # model = load_pretrained_model(model, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}.tar')
        # val_results = inference_binary_segmentation(model=model, test_loader=validation_loader,
        #                                             path=f"{run_path}/fold_{n}/", device=dev)
        # logging.info(val_results.mean())

        # results for test dataset
        logging.info(f"\n\n ###############  TESTING PHASE  ###############  \n\n")
        model = load_pretrained_model(model, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}.tar')
        with runtime_events.measure("test", setup="segmentation", fold=n) as event:
            with precision.autocast():
                test_results = inference_binary_segmentation(
                    model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev
                )
            event.update(examples=len(test_loader.dataset), batches=len(test_loader))
        logging.info(test_results.mean())

        end_time = time.perf_counter()
        logging.info(f"Total time for fold {n}: {end_time - fold_time:.2f}")

        # Clear the GPU memory after evaluating on the test data for this fold
        torch.cuda.empty_cache()


    # saving final results as a Excel file
    save_segmentation_results(run_path, n_splits=config_training['CV'])

    # Measuring total time
    end_time = time.perf_counter()
    logging.info(f"Total time for all of the folds: {end_time - init_time:.2f}")
    runtime_events.write()
    gpu_telemetry.stop()


if __name__ == "__main__":
    run()
