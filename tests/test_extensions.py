from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from stabilization.ablation import run_ablation
from stabilization.streaming import generate, transform_files
from stabilization.synthetic import SyntheticConfig


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ablation_preserves_scores_and_generalizes_to_held_out_items(self):
        result = run_ablation(SyntheticConfig(users=89, items=97, dim=8))
        self.assertEqual(result["held_out_items"], 48)
        self.assertLess(result["original_score_error"]["relative_l2"], 1e-12)
        for method in result["variants"].values():
            self.assertLess(method["within_version_score_error"]["relative_l2"], 1e-12)
        full = result["variants"]["full"]
        self.assertGreater(full["held_out_items_cosine"], 1 - 1e-12)
        self.assertLess(full["mixed_version_score_error"]["relative_l2"], 1e-12)
        procrustes = result["variants"]["procrustes"]
        self.assertGreater(procrustes["mixed_version_score_error"]["relative_l2"], 0.1)

    def test_disk_alignment_uses_ids_and_keeps_short_blocks(self):
        config = SyntheticConfig(users=43, items=35, dim=8)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            source = root / "source"
            generate(source, config, torch.float64, block_rows=5)
            previous, current = source / "version0", source / "version1"
            selected = np.arange(config.items)[1::2]
            items = np.load(current / "items.npy")[selected]
            np.save(current / "items.npy", items)
            # The last row has a new ID, so it is transformed but cannot be an anchor.
            current_ids = selected.copy()
            current_ids[-1] = config.items + 6
            np.save(current / "item_ids.npy", current_ids)

            reference = root / "reference"
            aligned = root / "aligned"
            transform_files(previous, reference, block_rows=3)
            report = transform_files(
                current, aligned, block_rows=3,
                reference_folder=previous, checkpoint_path=reference / "transforms.pt",
            )
            self.assertEqual(report["anchor_items"], len(selected) - 1)
            np.testing.assert_array_equal(np.load(aligned / "item_ids.npy"), current_ids)
            np.testing.assert_array_equal(np.load(aligned / "user_ids.npy"), np.arange(config.users))
            original_scores = np.load(current / "users.npy") @ items.T
            stable_scores = np.load(aligned / "users.npy") @ np.load(aligned / "items.npy").T
            np.testing.assert_allclose(stable_scores, original_scores, rtol=1e-11, atol=1e-11)

    def test_disk_alignment_rejects_missing_anchors_without_publishing(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            source = root / "source"
            generate(source, SyntheticConfig(users=15, items=14, dim=4), torch.float64, 5)
            previous, current = source / "version0", source / "version1"
            ids = np.load(current / "item_ids.npy") + 100
            np.save(current / "item_ids.npy", ids)
            reference = root / "reference"
            destination = root / "aligned"
            transform_files(previous, reference, 5)
            with self.assertRaisesRegex(ValueError, "No item IDs overlap"):
                transform_files(
                    current, destination, 5,
                    reference_folder=previous, checkpoint_path=reference / "transforms.pt",
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
