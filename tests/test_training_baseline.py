import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.training_baseline import MODES, dataset_clients, pooled_mapping


def _master(directory):
    rows = []
    # BUSI: two multi-task clients owning both tasks over the same images.
    for client, images in (("BUSI_mt_0", ("b0", "b1")), ("BUSI_mt_1", ("b2", "b3"))):
        for image in images:
            for task in ("seg", "cls"):
                rows.append({"fold": 0, "dataset": "BUSI", "client_id": client, "task": task,
                             "split": "train", "img_path": image, "mask_path": f"{image}_m",
                             "class": "benign"})
    # ISIC: disjoint pools; seg clients carry masks only, cls clients labels only.
    for client, images in (("ISIC_seg_0", ("s0", "s1")), ("ISIC_seg_1", ("s2",))):
        for image in images:
            rows.append({"fold": 0, "dataset": "ISIC", "client_id": client, "task": "seg",
                         "split": "train", "img_path": image, "mask_path": f"{image}_m",
                         "class": None})
    for client, images in (("ISIC_cls_0", ("c0",)), ("ISIC_cls_1", ("c1", "c2"))):
        for image in images:
            rows.append({"fold": 0, "dataset": "ISIC", "client_id": client, "task": "cls",
                         "split": "train", "img_path": image, "mask_path": None, "class": "NV"})
    rows.append({"fold": 0, "dataset": "ISIC", "client_id": "ISIC_seg_0", "task": "seg",
                 "split": "test", "img_path": "t0", "mask_path": "t0_m", "class": None})
    path = Path(directory) / "federated_mapping.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class PooledMappingTests(unittest.TestCase):
    def test_modes_map_to_tasks(self):
        self.assertEqual(MODES["segmentation"], ("seg",))
        self.assertEqual(MODES["multitask"], ("seg", "cls"))

    def test_clients_are_selected_by_dataset_and_task(self):
        with tempfile.TemporaryDirectory() as directory:
            master = _master(directory)
            self.assertEqual(dataset_clients(master, "ISIC", ("seg",)), ["ISIC_seg_0", "ISIC_seg_1"])
            self.assertEqual(dataset_clients(master, "BUSI", ("cls",)), ["BUSI_mt_0", "BUSI_mt_1"])

    def test_single_task_pool_is_the_union_of_client_slices(self):
        with tempfile.TemporaryDirectory() as directory:
            master = _master(directory)
            seg = pooled_mapping(master, 0, "BUSI", ("seg",), "train")
            self.assertEqual(sorted(seg.img_path), ["b0", "b1", "b2", "b3"])
            self.assertEqual(set(seg.task), {"seg"})
            cls = pooled_mapping(master, 0, "ISIC", ("cls",), "train")
            self.assertEqual(sorted(cls.img_path), ["c0", "c1", "c2"])

    def test_multitask_pool_has_one_row_per_image_with_granted_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            master = _master(directory)
            busi = pooled_mapping(master, 0, "BUSI", ("seg", "cls"), "train")
            self.assertEqual(len(busi), 4)
            self.assertTrue(busi.mask_path.notna().all() and busi["class"].notna().all())
            isic = pooled_mapping(master, 0, "ISIC", ("seg", "cls"), "train").set_index("img_path")
            self.assertEqual(len(isic), 6)
            # Disjoint pools: a seg image never gains a label and a cls image never gains a mask.
            self.assertTrue(isic.loc[["s0", "s1", "s2"], "class"].isna().all())
            self.assertTrue(isic.loc[["c0", "c1", "c2"], "mask_path"].isna().all())

    def test_test_rows_never_enter_the_training_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            master = _master(directory)
            pool = pooled_mapping(master, 0, "ISIC", ("seg",), "train")
            self.assertNotIn("t0", set(pool.img_path))

    def test_an_image_owned_by_two_clients_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            master = _master(directory)
            frame = pd.read_csv(master)
            duplicate = frame[frame.client_id == "ISIC_seg_1"].assign(client_id="ISIC_seg_0")
            pd.concat([frame, duplicate]).to_csv(master, index=False)
            with self.assertRaisesRegex(ValueError, "more than one client"):
                pooled_mapping(master, 0, "ISIC", ("seg",), "train")


if __name__ == "__main__":
    unittest.main()
