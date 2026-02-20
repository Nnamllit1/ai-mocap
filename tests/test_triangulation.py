import unittest

import numpy as np

from app.core.triangulation import TriangulationEngine
from app.models.config import TriangulationConfig
from app.services.calibration_store import CalibrationStore


class TriangulationTests(unittest.TestCase):
    def _store(self):
        store = CalibrationStore()
        k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        d = np.zeros((5, 1))
        store.intrinsics["cam0"] = (k, d)
        store.intrinsics["cam1"] = (k, d)
        store.intrinsics["cam2"] = (k, d)
        store.extrinsics["cam0"] = (np.eye(3), np.zeros((3, 1)))
        store.extrinsics["cam1"] = (np.eye(3), np.array([[0.25], [0.0], [0.0]]))
        store.extrinsics["cam2"] = (np.eye(3), np.array([[0.0], [0.2], [0.0]]))
        return store

    def _tri(self, **cfg_kwargs):
        cfg = TriangulationConfig(
            min_views=2,
            pair_conf_threshold=0.2,
            reproj_error_max=6.0,
            allow_single_view_fallback=True,
            single_view_conf_scale=0.55,
            single_view_max_age_ms=1000,
            **cfg_kwargs,
        )
        return TriangulationEngine(self._store(), cfg)

    def _tri_distorted(self, **cfg_kwargs):
        store = self._store()
        for cid, (k, _) in list(store.intrinsics.items()):
            store.intrinsics[cid] = (
                k,
                np.array([[-0.18], [0.05], [0.0], [0.0], [0.0]], dtype=np.float64),
            )
        cfg = TriangulationConfig(
            min_views=2,
            pair_conf_threshold=0.2,
            reproj_error_max=6.0,
            allow_single_view_fallback=True,
            single_view_conf_scale=0.55,
            single_view_max_age_ms=1000,
            **cfg_kwargs,
        )
        return TriangulationEngine(store, cfg)

    def _obs_for_xyz(self, tri: TriangulationEngine, xyz: np.ndarray, conf: float = 0.95):
        out = {}
        for cid in tri.projections:
            uv = tri._project(cid, xyz)
            out[cid] = (float(uv[0]), float(uv[1]), float(conf))
        return out

    def test_uses_available_tracking_cameras_and_ignores_missing(self):
        tri = self._tri()
        xyz_true = np.array([0.1, -0.05, 2.5], dtype=np.float64)
        obs = self._obs_for_xyz(tri, xyz_true)
        obs.pop("cam2")
        estimate = tri.estimate_joint(obs, timestamp=10.0)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.mode, "measured")
        self.assertEqual(set(estimate.inlier_camera_ids), {"cam0", "cam1"})
        np.testing.assert_allclose(estimate.xyz, xyz_true, atol=0.02, rtol=0)

    def test_discards_outlier_view_and_keeps_multiview_result(self):
        tri = self._tri(reproj_error_max=4.0)
        xyz_true = np.array([0.05, 0.03, 2.2], dtype=np.float64)
        obs = self._obs_for_xyz(tri, xyz_true)
        cam2 = obs["cam2"]
        obs["cam2"] = (cam2[0] + 140.0, cam2[1] - 110.0, cam2[2])
        estimate = tri.estimate_joint(obs, timestamp=10.0)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.mode, "measured")
        self.assertEqual(set(estimate.inlier_camera_ids), {"cam0", "cam1"})
        np.testing.assert_allclose(estimate.xyz, xyz_true, atol=0.04, rtol=0)

    def test_falls_back_to_single_view_when_exactly_one_valid_observation(self):
        tri = self._tri()
        prior_xyz = np.array([0.15, -0.08, 2.0], dtype=np.float64)
        obs = self._obs_for_xyz(tri, prior_xyz)
        single_obs = {"cam0": obs["cam0"]}
        estimate = tri.estimate_joint(
            single_obs,
            prior_xyz=prior_xyz,
            prior_timestamp=9.5,
            timestamp=10.0,
        )
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.mode, "single_view")
        self.assertEqual(estimate.inlier_camera_ids, ["cam0"])

    def test_falls_back_to_single_view_when_only_one_inlier_remains(self):
        tri = self._tri(reproj_error_max=2.0)
        prior_xyz = np.array([0.0, 0.0, 2.4], dtype=np.float64)
        obs = self._obs_for_xyz(tri, prior_xyz)
        obs["cam1"] = (obs["cam1"][0] + 160.0, obs["cam1"][1] + 160.0, 0.95)
        original_pair = tri._triangulate_pair

        # Force a candidate that is only consistent with cam0.
        def _forced_pair(*args, **kwargs):
            return np.array(prior_xyz, dtype=np.float64)

        tri._triangulate_pair = _forced_pair
        try:
            estimate = tri.estimate_joint(
                {"cam0": obs["cam0"], "cam1": obs["cam1"]},
                prior_xyz=prior_xyz,
                prior_timestamp=9.8,
                timestamp=10.0,
            )
        finally:
            tri._triangulate_pair = original_pair
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.mode, "single_view")
        self.assertEqual(estimate.inlier_camera_ids, ["cam0"])

    def test_distortion_aware_triangulation_recovers_point(self):
        tri = self._tri_distorted(reproj_error_max=5.0)
        xyz_true = np.array([0.12, -0.04, 2.8], dtype=np.float64)
        obs = self._obs_for_xyz(tri, xyz_true)
        estimate = tri.estimate_joint(obs, timestamp=20.0)
        self.assertIsNotNone(estimate)
        self.assertEqual(estimate.mode, "measured")
        self.assertGreaterEqual(len(estimate.inlier_camera_ids), 2)
        np.testing.assert_allclose(estimate.xyz, xyz_true, atol=0.05, rtol=0)

    def test_cheirality_marks_behind_camera_as_outlier(self):
        tri = self._tri()
        errors = tri._reprojection_errors(
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            {"cam0": (320.0, 240.0, 0.95)},
        )
        self.assertIn("cam0", errors)
        self.assertTrue(np.isinf(errors["cam0"]))


if __name__ == "__main__":
    unittest.main()
