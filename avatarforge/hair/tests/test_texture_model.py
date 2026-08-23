import unittest

from avatarforge_hair.texture_model import TEXTURE_LABELS, ProductState, apply_products, texture_state


class TextureModelTests(unittest.TestCase):
    def test_complete_editor_taxonomy(self):
        self.assertEqual(TEXTURE_LABELS, ("1A","1B","1C","2A","2B","2C","3A","3B","3C","4A","4B","4C"))

    def test_texture_space_tightens_monotonically(self):
        states = [texture_state(label) for label in TEXTURE_LABELS]
        self.assertEqual([s.curl_coordinate for s in states], sorted(s.curl_coordinate for s in states))
        self.assertEqual([s.rest_curvature_per_m for s in states], sorted(s.rest_curvature_per_m for s in states))
        self.assertEqual([s.rest_twist_per_m for s in states], sorted(s.rest_twist_per_m for s in states))
        self.assertEqual([s.visible_length_ratio for s in states], sorted((s.visible_length_ratio for s in states), reverse=True))

    def test_tightening_changes_more_than_amplitude(self):
        a = texture_state("2A")
        c = texture_state("4C")
        self.assertGreater(c.cycles_per_10cm, a.cycles_per_10cm * 5)
        self.assertLess(c.coil_radius_m, a.coil_radius_m)
        self.assertGreater(c.kink, a.kink)
        self.assertGreater(c.cross_plane, a.cross_plane)

    def test_product_response_is_bounded(self):
        state = texture_state("4C")
        compiled = apply_products(state, ProductState(water=1, conditioner=1, gel=1, oil=1))
        self.assertGreater(compiled["elongation"], 1)
        self.assertGreater(compiled["mass_scale"], 1)
        self.assertGreater(compiled["damping_scale"], 1)
        self.assertGreaterEqual(compiled["cohesion"], 0)
        self.assertLessEqual(compiled["cohesion"], 1)
        self.assertGreaterEqual(compiled["roughness"], 0.12)
        self.assertGreaterEqual(compiled["gloss"], 0)
        self.assertLessEqual(compiled["gloss"], 1)


if __name__ == "__main__":
    unittest.main()
