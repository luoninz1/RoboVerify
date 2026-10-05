import unittest
from unittest.mock import Mock

import numpy as np
from synthesis.api.instructions import Skip
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.region import BlockRegion
from synthesis.cfg.straightline import SearchBudget, straight_line_synthesize
from synthesis.cfg.synthesize import synthesize_cfg
from synthesis.mcmc.distance import KDEDistance, MMDDistance
from synthesis.mcmc.search_core import CandidatePool, acceptance_probability
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, ref


class SearchTests(unittest.TestCase):
    def test_recorded_and_scene_features_retain_identical_gripper_and_block_state(self):
        from synthesis.cfg.straightline import _features
        from synthesis.predicates.scene import scene_from_obs

        obs = np.arange(58, dtype=float)
        scenes = [scene_from_obs(obs, 3), scene_from_obs(obs + 0.1, 3)]
        np.testing.assert_array_equal(
            _features([obs, obs + 0.1], 3), _features(scenes, 3)
        )
        self.assertEqual(_features(scenes, 3).shape, (2, 14))

    def test_shared_acceptance_and_pool_filters_before_ranking(self):
        self.assertEqual(acceptance_probability(1, 0.1), 1.0)
        self.assertLess(
            acceptance_probability(-1, 0.1), acceptance_probability(-1, 1.0)
        )
        pool = CandidatePool(0.1, 10)
        for program, distance in [
            ("accurate", 0.2),
            ("goal", 0.25),
            ("bad_imitation", 5.0),
        ]:
            pool.add(program, distance)
        winner, score = pool.select(
            lambda p: {"accurate": 0, "goal": 1, "bad_imitation": 2}[p]
        )
        self.assertEqual(winner, "goal")
        pool.add("new_best", 0.0)
        self.assertEqual([e.program for e in pool.entries], ["new_best"])

    def test_cached_kl_is_deterministic_and_distinguishes_shift(self):
        data = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        distance = KDEDistance(data, seed=9)
        self.assertAlmostEqual(distance(data), 0.0)
        self.assertGreater(distance(data + 1), distance(data))
        self.assertEqual(distance(data + 0.1), distance(data + 0.1))

    def test_postscore_enabled_and_failed_task_not_called_success(self):
        scene = Scene({0: (0.0, 0.0, 0.425)}, {"b": 0})
        segment = DemoSegment(0, 0, 1, DemoTrace((scene, scene)), {"b": 0})
        candidate = Program(1, [Skip(0)])
        rollout = lambda p, s: list(s.states)
        result = straight_line_synthesize(
            [segment],
            boolean(True),
            candidate,
            lambda p, r: p,
            rollout=rollout,
            budget=SearchBudget(iterations=0, distance="mmd"),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.post_score, 1.0)
        failed = straight_line_synthesize(
            [segment],
            boolean(False),
            candidate,
            lambda p, r: p,
            rollout=rollout,
            budget=SearchBudget(iterations=1, cem_iterations=0, distance="mmd"),
        )
        self.assertFalse(failed.ok)

    def test_postscore_uses_recorded_loop_entry_instead_of_segment_start(self):
        def observation(stacked):
            obs = np.zeros(43)
            obs[10:13] = (0, 0, 0.425)
            obs[22:25] = (0, 0, 0.475) if stacked else (0.2, 0, 0.425)
            return obs

        bindings = {"a": 1, "b": 0}
        post = atom("ON_star_zero", ref("a"), ref("b"))
        candidate = Program(1, [Skip(0)])
        for as_scene in (False, True):
            for initially_stacked in (False, True):
                with self.subTest(
                    as_scene=as_scene, initially_stacked=initially_stacked
                ):
                    entry = observation(initially_stacked)
                    current = observation(not initially_stacked)
                    states = (current, entry, current, current)
                    if as_scene:
                        from synthesis.predicates.scene import scene_from_obs

                        states = tuple(scene_from_obs(s, 2, bindings) for s in states)
                    # Index 0 is outside this invocation, and index 2 is its later
                    # body segment: neither is the frozen loop-entry geometry.
                    segment = DemoSegment(
                        0,
                        2,
                        3,
                        DemoTrace(states, num_blocks=2),
                        bindings,
                        entry_index=1,
                    )
                    result = straight_line_synthesize(
                        [segment],
                        post,
                        candidate,
                        lambda p, r: p,
                        rollout=lambda p, s: list(s.states),
                        budget=SearchBudget(iterations=0, distance="mmd"),
                    )
                    self.assertEqual(result.distance, 0.0)
                    self.assertEqual(result.ok, initially_stacked)
                    self.assertEqual(result.post_score, float(initially_stacked))
                    if as_scene:
                        for key, position in states[2].positions.items():
                            np.testing.assert_array_equal(
                                states[2].entry_positions[key], position
                            )

    def test_failed_loop_body_is_refined_without_replacing_the_loop(self):
        from unittest.mock import patch

        import z3
        from synthesis.cfg.region import LoopRegion
        from synthesis.predicates.enumerate import SearchResult
        from synthesis.predicates.term import negate

        low = Scene({0: (0, 0, 0), 1: (0.2, 0, 0)}, {"a": 1, "b": 0})
        high = Scene({0: (0, 0, 0), 1: (0.2, 0, 0.1)}, low.bindings)
        placed = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, low.bindings)
        segment = DemoSegment(0, 0, 2, DemoTrace((low, high, placed)))
        post = atom("ON", ref("a"), ref("b"))
        body = RelationalCFG.initial([segment], boolean(True), post, ("a", "b"))
        loop = LoopRegion(
            boolean(True), (), (), invariant=z3.BoolVal(True), body_cfg=body
        )
        graph = RelationalCFG.initial([segment], boolean(True), post)
        graph.nodes["v0"].region = loop

        def realize(node, demos, post):
            if isinstance(node.region, LoopRegion):
                return node.region, True
            return BlockRegion((Skip(0),), (Skip(0),)), node.name != "v0"

        feature = negate(atom("Higher", ref("b"), ref("a")))
        with patch(
            "synthesis.cfg.refine.learn_classifier",
            return_value=SearchResult("found", feature),
        ):
            result = synthesize_cfg(
                graph, realize, lambda graph: [low], max_refinements=2
            )
        self.assertTrue(result)
        self.assertIs(graph.nodes["v0"].region, loop)
        self.assertEqual(loop.body_cfg.order, ["v0.0", "v0.1"])
        self.assertEqual(len(loop.body), 2)

    def test_driver_runs_all_blocks_and_retains_failure(self):
        cfg = RelationalCFG.initial([], boolean(True), boolean(True))
        region = BlockRegion((Skip(0),), (Skip(0),))
        result = synthesize_cfg(cfg, lambda n, d, p: (region, True), lambda g: [])
        self.assertTrue(result)
        self.assertIs(cfg.nodes["v0"].region, region)
        result = synthesize_cfg(
            cfg, lambda n, d, p: (region, False), lambda g: [], max_refinements=0
        )
        self.assertFalse(result)
        self.assertEqual(result.status, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
