import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import paths
from src.dataset.classic_dataloader import (classic_data_config, load_classic_datasets,
    class_weights_from_training, split_supervised_frame, supervision_frame)
from src.utils.segmentation_loss import DiceBCELoss
from src.utils.supervision import _combined_loss
from src.utils.metrics import segmentation_strata_metrics, summarize_segmentation_strata
from src.utils.experiment_init import init_criterion_segmentation
from src.experiments.study_runner import load_manifest, build_execution_plan, _config_signature, _partition_signature

ROOT = Path(__file__).resolve().parents[1]


class DiceBCETests(unittest.TestCase):
    def test_weighted_formula_and_legacy_dice(self):
        loss = DiceBCELoss(.3, .8)
        x = torch.randn(3, 1, 8, 8, requires_grad=True)
        y = torch.randint(0, 2, x.shape).float()
        y[0] = 0
        expected = .3 * init_criterion_segmentation('DICE')(x, y) + .8 * torch.nn.functional.binary_cross_entropy_with_logits(x, y)
        torch.testing.assert_close(loss(x, y), expected)
        loss(x, y).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_weights_and_multiclass_shape(self):
        for pair in ((-1, 1), (0, 0), (float('nan'), 1), (1, float('inf')), (True, 1)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                DiceBCELoss(*pair)
        with self.assertRaises(ValueError):
            DiceBCELoss()(torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8))

    def test_empty_target_has_corrective_gradient(self):
        x = torch.full((1, 1, 8, 8), 2., requires_grad=True)
        y = torch.zeros_like(x)
        loss = DiceBCELoss()(x, y)
        loss.backward()
        self.assertTrue((x.grad > 0).all())
        self.assertLess(DiceBCELoss()(x.detach()-x.grad, y), loss)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'BF16 CUDA required')
    def test_bf16_mixed_targets(self):
        x = torch.randn(2, 1, 8, 8, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        y = torch.zeros_like(x)
        y[1, :, 2:5, 2:5] = 1
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = DiceBCELoss().cuda()(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_missing_placeholders_never_affect_losses_or_gradients(self):
        data = dict(image=torch.zeros(3, 1, 4, 4), mask=torch.zeros(3, 1, 4, 4),
                    label=torch.tensor([[0.], [-1.], [1.]]),
                    has_mask=torch.tensor([False, True, True]), has_label=torch.tensor([True, False, True]))
        def evaluate(data):
            seg = torch.zeros(3, 1, 4, 4, requires_grad=True)
            cls = torch.zeros(3, 3, requires_grad=True)
            loss, counted, _ = _combined_loss(data, [cls], [seg], ('seg', 'cls'),
                {'seg': .8, 'cls': .2}, 'cpu', 3, DiceBCELoss(), torch.nn.CrossEntropyLoss(), True)
            loss.backward()
            return loss.detach(), seg.grad, cls.grad
        original = evaluate(data)
        changed = copy.deepcopy(data)
        changed['mask'][0] = float('nan')
        changed['label'][1] = 999
        for a, b in zip(original, evaluate(changed)):
            torch.testing.assert_close(a, b)
        self.assertEqual(float(original[1][0].abs().sum()), 0.)
        self.assertEqual(float(original[2][1].abs().sum()), 0.)

    def test_absent_task_keeps_its_companions_weight(self):
        data = dict(image=torch.zeros(2, 1, 4, 4), mask=torch.zeros(2, 1, 4, 4),
                    label=torch.tensor([[0.], [1.]]), has_mask=torch.zeros(2, dtype=torch.bool),
                    has_label=torch.ones(2, dtype=torch.bool))
        logits = torch.zeros(2, 1, requires_grad=True)
        loss, counted, _ = _combined_loss(data, logits, torch.zeros_like(data['mask']), ('seg', 'cls'),
                {'seg': .8, 'cls': .2}, 'cpu', 2, DiceBCELoss(), torch.nn.BCEWithLogitsLoss(), True)
        torch.testing.assert_close(loss, .2*torch.nn.functional.binary_cross_entropy_with_logits(logits, data['label']))
        self.assertEqual(counted, ['cls'])
        data['has_label'][0] = False
        with self.assertRaisesRegex(ValueError, 'Each image'):
            _combined_loss(data, logits, data['mask'], ('seg', 'cls'), {'seg': .8, 'cls': .2},
                           'cpu', 2, DiceBCELoss(), torch.nn.BCEWithLogitsLoss(), True)


class StrataMetricsTests(unittest.TestCase):
    def test_positive_empty_and_missing_strata(self):
        empty = np.zeros((1, 1, 4, 4))
        positive = empty.copy(); positive[0, 0, 0, 0] = 1
        rows = [segmentation_strata_metrics(empty, empty), segmentation_strata_metrics(empty, positive),
                segmentation_strata_metrics(positive, empty), segmentation_strata_metrics(positive, positive)]
        result = summarize_segmentation_strata(rows)
        self.assertEqual(result['dice_positive'], .5)
        self.assertEqual(result['iou_positive'], .5)
        self.assertEqual(result['empty_fp_image_rate'], .5)
        self.assertEqual(result['empty_predicted_area_fraction'], 1/32)
        self.assertEqual((result['n_positive'], result['n_empty']), (2, 2))
        self.assertTrue(np.isnan(summarize_segmentation_strata(rows[:2])['dice_positive']))
        self.assertTrue(np.isnan(summarize_segmentation_strata(rows[2:])['empty_fp_image_rate']))


class ClassicDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((ROOT/'src/config.yaml').read_text())

    def test_busi_normals_are_supervised_and_weights_use_training_only(self):
        cfg = copy.deepcopy(self.config)
        data = classic_data_config(cfg)
        loaders = load_classic_datasets(cfg['training'], data, None, ('seg','cls'))
        train, val, test = [group[0].dataset for group in loaders]
        normal = train.mapping_file[train.mapping_file['class'] == 'normal']
        self.assertGreater(len(normal), 0)
        for index in normal.index[:3]:
            row = train[int(index)]
            self.assertTrue(row['has_mask']); self.assertTrue(row['has_label'])
            self.assertEqual(row['mask'].sum(), 0)
        self.assertEqual(train.class_weights, class_weights_from_training(train.mapping_file, data['classes']))
        for other in (val, test):
            self.assertFalse(set(train.mapping_file.img_path) & set(other.mapping_file.img_path))

    def test_isic_disjoint_pools_and_groups_survive_classic_split(self):
        cfg = copy.deepcopy(self.config); cfg['data']['dataset'] = 'ISIC_2018'
        data = classic_data_config(cfg)
        loaders = load_classic_datasets(cfg['training'], data, None, ('seg','cls'))
        frames = [x[0].dataset.mapping_file for x in loaders]
        self.assertEqual(loaders[0][0].dataset.channels, 3)
        self.assertEqual(len(loaders[0][0].dataset.classes), 7)
        for i, frame in enumerate(frames):
            self.assertTrue(frame.mask_path.notna().any()); self.assertTrue(frame['class'].notna().any())
            for prior in frames[:i]:
                self.assertFalse(set(frame.img_path) & set(prior.img_path))
                self.assertFalse(set(frame.lesion_id.dropna()) & set(prior.lesion_id.dropna()))

    def test_cross_pool_groups_and_unsupervised_images_rejected(self):
        frame = pd.DataFrame(dict(img_path=['a','b'], mask_path=['m',None],
                                  **{'class':[None,'benign']}, lesion_id=['g','g']))
        with self.assertRaisesRegex(ValueError, 'crosses'):
            split_supervised_frame(frame, self.config['training'], self.config['data'])
        frame.loc[0, 'mask_path'] = None
        with self.assertRaisesRegex(ValueError, 'without mask'):
            supervision_frame(frame, self.config['data'], ('seg','cls'))

    def test_multitask_oversampling_is_refused(self):
        data = copy.deepcopy(self.config['data']); data['oversampling'] = True
        with self.assertRaisesRegex(ValueError, 'oversampling'):
            load_classic_datasets(self.config['training'], data, None, ('seg','cls'))

    def test_new_protocol_is_pinned_and_hash_sensitive(self):
        manifest = load_manifest(ROOT/'studies/example_multi_dataset.yaml')
        row = build_execution_plan(manifest, [1993], manifest['arms'][:1])[0][0]
        config = row['config']
        self.assertIn('example_multi_dataset_dice_bce', row['partition_path'])
        self.assertEqual(config['loss']['function'], 'DiceBCE')
        self.assertEqual(config['datasets']['Curated_BUSI']['seg_exclude_classes'], [])
        changed = copy.deepcopy(config); changed['loss']['bce_weight'] = .8
        self.assertNotEqual(_config_signature(config), _config_signature(changed))
        self.assertEqual(_partition_signature(config), _partition_signature(changed))
        changed['datasets']['Curated_BUSI']['seg_exclude_classes'] = ['normal']
        self.assertNotEqual(_partition_signature(config), _partition_signature(changed))




class _PartialImages(torch.utils.data.Dataset):
    classes = ['a', 'b', 'c']

    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.full((1, 8, 8), float(index % 2))
        mask = image.clone()
        if index == 0:
            mask[:] = float('nan')  # deliberately invalid placeholder, never consumed
        return dict(image=image, mask=mask, label=torch.tensor([-1. if index == 1 else float(index % 3)]),
                    has_mask=torch.tensor(index != 0), has_label=torch.tensor(index != 1),
                    patient_id=index, **{'class': '' if index == 1 else self.classes[index % 3]})


class _TinyMultitask(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, image):
        seg = self.scale * (image * 2 - 1)
        value = image.mean((2, 3)) * self.scale
        return [torch.cat([value, -value, value*0], dim=1)], [seg]


class ClassicExecutionTests(unittest.TestCase):
    def test_partial_epoch_and_inference_batch_invariance(self):
        from src.utils.classic_multitask import run_epoch, inference_supervised_multitask
        from src.utils.training_runtime import PrecisionPolicy
        from src.federated.unified_eval import evaluate
        policy = PrecisionPolicy('fp32', 'cpu')
        model = _TinyMultitask()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        result = run_epoch(model, DataLoader(_PartialImages(), batch_size=1), 'cpu', policy,
                           3, DiceBCELoss(), torch.nn.CrossEntropyLoss(), .8, True, optimizer)
        self.assertEqual(result['seg_samples'], 3)
        self.assertEqual(result['cls_samples'], 3)
        self.assertTrue(np.isfinite(result['loss']))
        self.assertEqual(result['n_positive'], 2)
        self.assertEqual(result['n_empty'], 1)
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            metrics = []
            for batch in (1, 4):
                loader = DataLoader(_PartialImages(), batch_size=batch)
                frames.append(inference_supervised_multitask(model, loader, Path(directory)/str(batch), 'cpu', policy))
                seg = evaluate(model, loader, 'seg', 3, 'cpu')[0]
                cls = evaluate(model, loader, 'cls', 3, 'cpu')[0]
                self.assertEqual(seg['n_test'], 3)
                self.assertEqual(cls['n_test'], 3)
                metrics.append((seg, cls))
            for task in range(2):
                pd.testing.assert_frame_equal(frames[0][task], frames[1][task])
                for key in metrics[0][task]:
                    np.testing.assert_allclose(metrics[0][task][key], metrics[1][task][key], equal_nan=True)

    def test_busi_single_task_preserves_the_historical_stratified_split(self):
        from src.dataset.BUSI_dataloader import BUSI_dataloader_CV
        config = yaml.safe_load((ROOT/'src/config.yaml').read_text())
        data = classic_data_config(config)
        old = BUSI_dataloader_CV(seed=config['training']['seed'], batch_size=data['batch_size'],
                 transforms=None, classes=data['classes'], n_folds=1, oversampling=False,
                 path_images=paths.processed_dir(data), train_size=data['train_size'])
        new = load_classic_datasets(config['training'], data, None, ('seg',))
        for previous, current in zip(old, new):
            self.assertEqual(set(previous[0].dataset.mapping_file.img_path),
                             set(current[0].dataset.mapping_file.img_path))

    def test_primary_metric_and_negative_metrics_reach_reports(self):
        from src.experiments.analyze import summary_table, summary_by_seed, per_client_deltas
        from src.experiments.executive_report import _results_table, _key_findings
        rows = [dict(setup=setup, dataset='Curated_BUSI', fold=0, client_id='c', task='seg',
                     dice=.9, dice_positive=dice, iou_positive=.5, empty_fp_image_rate=np.nan,
                     empty_predicted_area_fraction=np.nan, n_positive=3, n_empty=0,
                     segmentation_primary_metric='dice_positive', evaluation_scheme='holdout')
                for setup, dice in [('federated', .6), ('standalone', .4)]]
        frame = pd.DataFrame(rows)
        summary = summary_table(frame)
        self.assertIn('empty_fp_image_rate', set(summary.metric))
        self.assertIn('empty_fp_image_rate', set(summary_by_seed(frame).metric))
        self.assertEqual(set(per_client_deltas(frame).metric), {'dice_positive'})
        self.assertAlmostEqual(per_client_deltas(frame).delta.iloc[0], .2)
        self.assertIn('Falsos positivos', _results_table(summary))
        self.assertNotIn('maior efeito', _key_findings(pd.DataFrame()))
        # Old CSVs keep their original primary outcome.
        legacy = frame.drop(columns='segmentation_primary_metric')
        self.assertEqual(set(per_client_deltas(legacy).metric), {'dice'})


if __name__ == '__main__':
    unittest.main()
