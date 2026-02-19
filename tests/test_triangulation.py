import unittest

import numpy as np

from app.core.triangulation import TriangulationEngine
from app.models.config import TriangulationConfig
from app.services.calibration_store import CalibrationStore


class TriangulationTests(unittest.TestCase):
    def test_pairwise_triangulation(self):
        store = CalibrationStore()
        k = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]])
        d = np.zeros((5, 1))
        store.intrinsics["cam0"] = (k, d)
        store.intrinsics["cam1"] = (k, d)
        store.extrinsics["cam0"] = (np.eye(3), np.zeros((3, 1)))
        store.extrinsics["cam1"] = (np.eye(3), np.array([[0.2], [0.0], [0.0]]))
        tri = TriangulationEngine(store, TriangulationConfig(min_views=2))
        observations = {"cam0": (160.0, 120.0, 1.0), "cam1": (170.0, 120.0, 1.0)}
        xyz = tri.triangulate_joint(observations)
        self.assertIsNotNone(xyz)


if __name__ == "__main__":
    unittest.main()
