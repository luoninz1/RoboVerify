"""Physical table geometry is separate from relational and observation state."""

import unittest

from synthesis.mcmc.synthesis import make_roboverify_env


class TableGeometry(unittest.TestCase):
    def test_tower_tasks_expose_the_table_plane_without_changing_observations(self):
        for task in ("stack", "unstack", "reverse", "partial"):
            with self.subTest(task=task):
                env = make_roboverify_env(task=task, num_blocks=4)
                try:
                    obs, _ = env.reset(seed=42)
                    inner = env.env
                    self.assertEqual(inner.table_surface_height, 0.4)
                    self.assertEqual(
                        inner.table_surface_height,
                        float(inner.sim.data.get_geom_xpos("table0")[2]),
                    )
                    self.assertGreater(inner.height_offset, inner.table_surface_height)
                    self.assertEqual((inner.agent_dim, inner.object_dyn_dim), (10, 12))
                    self.assertEqual(len(obs), 10 + 12 * 4 + 3 * 4 + 3)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
