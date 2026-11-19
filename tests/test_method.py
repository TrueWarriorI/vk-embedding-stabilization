import unittest

import torch

from stabilization.method import align, fit_svd, procrustes
from stabilization.metrics import mean_cosine, score_error
from stabilization.synthetic import SyntheticConfig, make_embeddings, plane_rotation
from stabilization.tsqr import local_r, tsqr_r


class StabilizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = SyntheticConfig(users=73, items=55, dim=8)
        self.users, self.items = make_embeddings(self.config, version=0)

    def test_short_and_empty_qr_blocks(self):
        for rows in (0, 3, 55):
            with self.subTest(rows=rows):
                block = self.items[:rows]
                factor = local_r(block)
                self.assertEqual(factor.shape, (8, 8))
                torch.testing.assert_close(factor.T @ factor, block.T @ block)
        factor = tsqr_r(self.items)
        torch.testing.assert_close(factor.T @ factor, self.items.T @ self.items)

    def test_reduction_matches_dense_qr(self):
        factors = [local_r(shard) for shard in self.items.tensor_split(4)]
        merged = [local_r(torch.cat(factors[k : k + 2])) for k in (0, 2)]
        root = local_r(torch.cat(merged))
        direct = torch.linalg.qr(self.items, mode="r").R
        torch.testing.assert_close(root.T @ root, direct.T @ direct)

    def test_score_svd_matches_dense_baseline(self):
        transforms = fit_svd(self.users, self.items)
        stable_users, stable_items = transforms.apply(self.users, self.items)
        scores = self.items @ self.users.T
        torch.testing.assert_close(stable_items @ stable_users.T, scores, atol=1e-11, rtol=1e-11)
        expected = torch.linalg.svdvals(scores)[: self.config.dim]
        torch.testing.assert_close(transforms.singular_values, expected)

    def test_procrustes_uses_row_orientation(self):
        rotation = plane_rotation(8, 57, self.items.dtype, self.items.device)
        recovered = procrustes(self.items @ rotation, self.items)
        torch.testing.assert_close(recovered, rotation.T)
        torch.testing.assert_close(recovered.T @ recovered, torch.eye(8, dtype=self.items.dtype))

    def test_rotation_recovery_and_metrics(self):
        users1, items1 = make_embeddings(self.config, version=1)
        ref_users, ref_items = fit_svd(self.users, self.items).apply(self.users, self.items)
        current = align(fit_svd(users1, items1), items1, ref_items)
        stable_users, stable_items = current.apply(users1, items1)
        self.assertLess(mean_cosine(self.users, users1)["mean"], 0.4)
        self.assertAlmostEqual(mean_cosine(ref_users, stable_users)["mean"], 1, places=12)
        self.assertAlmostEqual(mean_cosine(ref_items, stable_items)["mean"], 1, places=12)
        self.assertLess(score_error(users1, items1, stable_users, stable_items)["relative_l2"], 1e-12)

    def test_rank_deficient_and_zero_scores(self):
        for scale, expected_rank in ((1.0, 4), (0.0, 0)):
            with self.subTest(scale=scale):
                users = self.users.clone()
                users[:, 4:] = users[:, :4] * 2
                items = self.items * scale
                transforms = fit_svd(users, items)
                stable_users, stable_items = transforms.apply(users, items)
                self.assertEqual(transforms.effective_rank, expected_rank)
                torch.testing.assert_close(stable_items @ stable_users.T, items @ users.T)
                self.assertTrue(torch.isfinite(stable_users).all())
        self.assertIsNone(mean_cosine(stable_users, stable_users)["mean"])

    def test_synthetic_rows_are_independent_of_process_count(self):
        config = SyntheticConfig(users=73, items=55, dim=8, noise=0.15)
        for version in (0, 1):
            full = make_embeddings(config, version=version)
            shards = [make_embeddings(config, version=version, rank=k, size=4) for k in range(4)]
            for expected, parts in zip(full, zip(*shards), strict=True):
                torch.testing.assert_close(torch.cat(parts), expected)


if __name__ == "__main__":
    unittest.main()
