import unittest
import numpy as np

from avatarforge_hair.micro_fiber import microstate_axis, integrate_rest_shape, curve_metrics, apply_environment


class MicroFiberTests(unittest.TestCase):
    def test_geometry_emerges_from_microstate(self):
        straight = curve_metrics(integrate_rest_shape(microstate_axis(0.05), length_m=0.06))
        coily = curve_metrics(integrate_rest_shape(microstate_axis(0.95), length_m=0.06))
        self.assertGreater(coily['median_curvature_per_m'], straight['median_curvature_per_m'] * 8)
        self.assertLess(coily['visible_length_ratio'], straight['visible_length_ratio'])

    def test_water_changes_microstate_then_geometry(self):
        dry = microstate_axis(0.85)
        wet = apply_environment(dry, water=1.0)
        dry_m = curve_metrics(integrate_rest_shape(dry, length_m=0.05))
        wet_m = curve_metrics(integrate_rest_shape(wet, length_m=0.05))
        self.assertLess(wet.cortical_strain_bias, dry.cortical_strain_bias)
        self.assertNotAlmostEqual(wet_m['median_curvature_per_m'], dry_m['median_curvature_per_m'])

    def test_root_is_fixed(self):
        p = integrate_rest_shape(microstate_axis(0.7), length_m=0.02)
        self.assertTrue(np.allclose(p[0], 0.0))


if __name__ == '__main__':
    unittest.main()
