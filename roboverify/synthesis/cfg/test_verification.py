import unittest

from synthesis.cfg.demos import DemoAssignment
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.region import BlockRegion
from synthesis.cfg.verification import propose_summaries, verify_cfg_motion
from synthesis.predicates.term import (
    atom,
    boolean,
    conjunction,
    disjunction,
    forall,
    negate,
    ref,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.test_primitive_motion import primitives


def placement_cfg(split=False):
    a, b, b0, x = map(ref, ("a", "b", "b0", "x"))
    pre = conjunction(
        forall(["x"], disjunction(atom("eq", x, a), atom("eq", x, b))),
        atom("eq", b, b0),
        atom("Scattered", a, b),
        atom("Higher", b, a),
        negate(atom("Higher", a, b)),
    )
    post = atom("ON_star", a, b)
    cfg = RelationalCFG.initial([], pre, post, ("a", "b", "b0"))
    if split:
        cfg.nodes = {
            "v0": Node("v0", BlockRegion(None, tuple(primitives()[:2]))),
            "v1": Node("v1", BlockRegion(None, tuple(primitives()[2:]))),
        }
        cfg.order = ["v0", "v1"]
        cfg.edges = [
            Edge("entry", "v0", pre),
            Edge("v0", "v1", boolean(True)),
            Edge("v1", "exit", post),
        ]
    else:
        cfg.nodes["v0"].region = BlockRegion(None, tuple(primitives()))
    return cfg


class CFGVerificationTests(unittest.TestCase):
    def verify(self, cfg):
        context = HighLevelContext()
        propose_summaries(cfg, context)
        return verify_cfg_motion(
            cfg,
            context,
            initial_positions={"a": [0.3, 0, -0.1], "b": [0, 0, 0], "b0": [0, 0, 0]},
            initial_arm=[0.3, 0, 0.2],
        )

    def test_actual_primitive_candidate_is_verified(self):
        result = self.verify(placement_cfg())
        self.assertTrue(result, str(result))

    def test_transfer_across_cfg_blocks_uses_composed_state(self):
        cfg = placement_cfg(split=True)
        result = self.verify(cfg)
        self.assertTrue(result, str(result))
        self.assertEqual(result.checked_blocks, 2)
        self.assertEqual(type(cfg.nodes["v0"].region.symbolic[0]).__name__, "Skip")
        self.assertEqual(type(cfg.nodes["v1"].region.symbolic[0]).__name__, "Put")

    def test_root_selection_ignores_unrelated_b0_in_integrated_verifier(self):
        from synthesis.predicates.term import implies

        a, b, decoy, base, u = map(ref, ("a", "b", "b0", "z_base", "u"))
        pre = conjunction(
            forall(
                ["u"], disjunction(*(atom("eq", u, n) for n in (a, b, decoy, base)))
            ),
            forall(["u"], implies(atom("ON_star", base, u), atom("eq", u, base))),
            atom("ON_star", b, base),
            negate(atom("eq", b, base)),
            atom("Scattered", a, b),
            atom("Scattered", a, base),
            atom("Scattered", decoy, b),
            atom("Scattered", decoy, base),
            atom("Scattered", decoy, a),
            atom("Higher", base, a),
            negate(atom("Higher", a, base)),
        )
        cfg = RelationalCFG.initial(
            [], pre, atom("ON_star", a, b), ("a", "b", "b0", "z_base")
        )
        cfg.nodes["v0"].region = BlockRegion(None, tuple(primitives()))
        ctx = HighLevelContext()
        propose_summaries(cfg, ctx)
        result = verify_cfg_motion(
            cfg,
            ctx,
            initial_positions={
                "a": [0.3, 0, -0.1],
                "b": [0, 0, 0.05],
                "z_base": [0, 0, 0],
                "b0": [2, 2, 0],
            },
            initial_arm=[0.3, 0, 0.2],
        )
        self.assertTrue(result, str(result))
        roots = [c for c in result.checks if c.obligation.endswith("/root_selection")]
        self.assertEqual(len(roots), 1)
        self.assertIn("root=z_base;", roots[0].reason)
        self.assertTrue(
            any(
                c.obligation.endswith("/alignment") and c.status == "valid"
                for c in result.checks
            )
        )

    def test_fresh_loop_context_retains_input_alignment_and_checks_new_placement(self):
        from synthesis.api.instructions import MoveByName
        from synthesis.cfg.region import LoopRegion
        from synthesis.predicates.term import implies, to_z3

        ctx = HighLevelContext()
        a, b, b0, u = map(ref, ("a", "b", "b0", "u"))
        invariant = conjunction(
            forall(["u"], disjunction(atom("eq", u, a), atom("eq", u, b))),
            atom("eq", b, b0),
            negate(atom("eq", a, b)),
            forall(["u"], implies(atom("ON_star", b, u), atom("eq", u, b))),
        )
        guard = conjunction(
            atom("Scattered", a, b), atom("Higher", b, a), negate(atom("Higher", a, b))
        )
        for displacement, expected in ((0.0, "valid"), (0.015, "refuted")):
            with self.subTest(displacement=displacement):
                body = placement_cfg()
                instructions = list(body.nodes["v0"].region.physical)
                instructions[-2] = MoveByName(
                    "b", "b", "b", target_offset=[displacement, 0, 0.05]
                )
                body.nodes["v0"].region = BlockRegion(None, tuple(instructions))
                cfg = RelationalCFG.initial(
                    [], conjunction(invariant, guard), boolean(True), ("a", "b", "b0")
                )
                cfg.nodes["v0"].region = LoopRegion(
                    guard,
                    (),
                    (body.nodes["v0"].region,),
                    invariant=to_z3(invariant, ctx),
                    body_cfg=body,
                )
                propose_summaries(cfg, ctx)
                result = verify_cfg_motion(cfg, ctx)
                roots = [
                    c for c in result.checks if c.obligation.endswith("/root_selection")
                ]
                self.assertEqual([c.status for c in roots], ["valid"])
                alignment = [
                    c for c in result.checks if c.obligation.endswith("/alignment")
                ]
                self.assertEqual([c.status for c in alignment], [expected])

    def test_unsupported_summary_is_not_replaced_with_skip(self):
        cfg = placement_cfg()
        cfg.edges[-1] = Edge("v0", "exit", boolean(True))
        from synthesis.api.instructions import PickByName, ReleaseByName

        cfg.nodes["v0"].region.physical = (
            PickByName("a"),
            ReleaseByName("a", target_z=0.1),
        )
        with self.assertRaisesRegex(ValueError, "explicit ON"):
            propose_summaries(cfg, HighLevelContext())


class VerifiedSynthesisTests(unittest.TestCase):
    def test_synthesized_candidate_reaches_both_real_verifiers(self):
        from synthesis.cfg.verified_synthesis import verified_synthesis

        cfg = placement_cfg()
        body = cfg.nodes["v0"].region.physical
        cfg.nodes["v0"].region = None
        seen = []

        def realize(node, demos, post):
            seen.append(node.name)
            return BlockRegion(None, body), True

        result = verified_synthesis(
            cfg,
            realize,
            lambda _: [],
            HighLevelContext(),
            symbolic_iterations=0,
            motion_iterations=0,
            min_blocks=2,
            max_blocks=2,
            motion_options={
                "initial_positions": {
                    "a": [0.3, 0, -0.1],
                    "b": [0, 0, 0],
                    "b0": [0, 0, 0],
                },
                "initial_arm": [0.3, 0, 0.2],
            },
        )
        self.assertTrue(
            bool(result),
            f"{result.status}: {result.reason}; {result.symbolic}; {result.motion}",
        )
        self.assertEqual(seen, ["v0"])
        self.assertEqual(result.symbolic.scope, "unbounded")
        self.assertEqual(
            [h["stage"] for h in result.history], ["synthesis", "symbolic", "motion"]
        )

    def test_supplied_candidate_skips_search_and_reaches_both_verifiers(self):
        from unittest.mock import Mock

        from synthesis.cfg.verified_synthesis import verified_synthesis

        cfg = placement_cfg()
        realize = Mock(side_effect=AssertionError("Initial search must be skipped"))
        prepare = Mock(return_value={})
        result = verified_synthesis(
            cfg,
            realize,
            lambda _: [],
            HighLevelContext(),
            initial_candidate=True,
            prepare=prepare,
            symbolic_iterations=0,
            motion_iterations=0,
            min_blocks=2,
            max_blocks=2,
            motion_options={
                "initial_positions": {
                    "a": [0.3, 0, -0.1],
                    "b": [0, 0, 0],
                    "b0": [0, 0, 0],
                },
                "initial_arm": [0.3, 0, 0.2],
            },
        )
        self.assertTrue(result, result.reason)
        realize.assert_not_called()
        prepare.assert_called_once_with(cfg, 0)
        self.assertTrue(result.symbolic)
        self.assertTrue(result.motion)

    def test_symbolic_failure_exports_a_concrete_demo_request(self):
        from synthesis.api.instructions import Skip
        from synthesis.cfg.verified_synthesis import verified_synthesis

        cfg = placement_cfg()
        result = verified_synthesis(
            cfg,
            lambda *_: (BlockRegion(None, (Skip(0),)), True),
            lambda _: [],
            HighLevelContext(),
            min_blocks=2,
            max_blocks=2,
        )
        self.assertEqual(result.status, "needs_demonstrations")
        self.assertEqual(result.request["kind"], "body")
        self.assertIsNotNone(result.request["configuration"])
        self.assertIsNone(result.motion)

    def test_motion_repair_reuses_counterexamples_and_preserves_summary(self):
        from copy import deepcopy

        from synthesis.api.program import Program
        from synthesis.cfg.verified_synthesis import verified_synthesis

        cfg = placement_cfg()
        cfg.nodes["v0"].region.physical = tuple(primitives(0.1))
        penalized = []
        revisions = []

        def repair(node, demos, post, penalty):
            bad = Program(len(node.region.physical), list(node.region.physical))
            from synthesis.api.instructions import MoveByName

            changed_structure = primitives()
            changed_structure.insert(
                1, MoveByName("a", "a", "a", target_offset=[0, 0, 0.01])
            )
            good = Program(len(changed_structure), changed_structure)
            penalized.append((penalty(bad), penalty(good)))
            return BlockRegion(node.region.symbolic, tuple(good.instructions))

        result = verified_synthesis(
            cfg,
            lambda n, *_: (n.region, True),
            lambda _: [],
            HighLevelContext(),
            repair_motion=repair,
            prepare=lambda cfg, revision: revisions.append(revision) or {},
            min_blocks=2,
            max_blocks=2,
            motion_options={
                "initial_positions": {
                    "a": [0.3, 0, -0.1],
                    "b": [0, 0, 0],
                    "b0": [0, 0, 0],
                },
                "initial_arm": [0.3, 0, 0.2],
            },
        )
        self.assertTrue(
            bool(result), f"{result.status}: {result.reason}; {result.motion}"
        )
        self.assertEqual(len(penalized), 1)
        self.assertEqual(revisions, [0, 1])
        self.assertEqual(len(result.cfg.nodes["v0"].region.physical), 6)
        self.assertGreater(penalized[0][0], 0)
        self.assertEqual(penalized[0][1], 0)
        self.assertEqual([h["stage"] for h in result.history].count("symbolic"), 2)
        self.assertEqual([h["stage"] for h in result.history].count("motion"), 2)

    def test_motion_repair_cannot_change_get_bindings(self):
        from synthesis.api.instructions import Assign
        from synthesis.cfg.verified_synthesis import verified_synthesis

        cfg = placement_cfg()
        cfg.nodes["v0"].region.physical = tuple(primitives(0.1))

        def repair(node, *args):
            return BlockRegion(
                node.region.symbolic, (Assign("new_alias", "a"), *primitives())
            )

        result = verified_synthesis(
            cfg,
            lambda n, *_: (n.region, True),
            lambda _: [],
            HighLevelContext(),
            repair_motion=repair,
            min_blocks=2,
            max_blocks=2,
            motion_options={
                "initial_positions": {
                    "a": [0.3, 0, -0.1],
                    "b": [0, 0, 0],
                    "b0": [0, 0, 0],
                },
                "initial_arm": [0.3, 0, 0.2],
            },
        )
        self.assertEqual(result.status, "invalid_motion_repair")

    def test_new_demonstrations_trigger_resynthesis_then_verification(self):
        from synthesis.api.instructions import Skip
        from synthesis.cfg.demos import DemoSegment, DemoTrace
        from synthesis.cfg.verified_synthesis import verified_synthesis
        from synthesis.predicates.scene import Scene

        cfg = placement_cfg()
        aliases = {"a": 0, "b": 1, "b0": 1}
        before = Scene({0: (0.3, 0, -0.1), 1: (0, 0, 0)}, aliases)
        after = Scene({0: (0, 0, 0.05), 1: (0, 0, 0)}, aliases)
        demo = DemoSegment(0, 0, 1, DemoTrace((before, after), num_blocks=2), aliases)
        cfg.demos.segments["v0"] = [demo]
        attempts, requests = [], []

        def realize(node, demos, post):
            attempts.append(len(demos))
            return (
                BlockRegion(
                    None, tuple(primitives()) if len(attempts) > 1 else (Skip(0),)
                ),
                True,
            )

        def provide(request):
            requests.append(request)
            return [demo]

        result = verified_synthesis(
            cfg,
            realize,
            lambda _: [],
            HighLevelContext(),
            demo_provider=provide,
            min_blocks=2,
            max_blocks=2,
            motion_options={
                "initial_positions": {
                    "a": [0.3, 0, -0.1],
                    "b": [0, 0, 0],
                    "b0": [0, 0, 0],
                },
                "initial_arm": [0.3, 0, 0.2],
            },
        )
        self.assertTrue(bool(result), f"{result.status}: {result.reason}")
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(len(requests), 1)

    def test_preservation_counterexample_relearns_invariant(self):
        from synthesis.api.instructions import Skip
        from synthesis.cfg.demos import DemoSegment, DemoTrace
        from synthesis.cfg.region import LoopRegion
        from synthesis.cfg.verified_synthesis import verified_synthesis
        from synthesis.predicates.scene import Scene
        from synthesis.predicates.term import implies, to_z3

        ctx = HighLevelContext()
        a, b, c, x, y = map(ref, ("a", "b", "c", "x", "y"))
        facts = conjunction(
            forall(["x"], disjunction(atom("eq", x, a), atom("eq", x, c))),
            negate(atom("eq", a, c)),
            forall(["x", "y"], atom("Higher", x, y)),
            forall(
                ["x", "y"], implies(negate(atom("eq", x, y)), atom("Scattered", x, y))
            ),
        )
        pre = conjunction(facts, atom("eq", b, a))
        aliases = {"a": 0, "b": 0, "c": 1}
        scene = Scene({0: (0, 0, 0), 1: (0.2, 0, 0)}, aliases)
        head = DemoSegment(0, 0, 1, DemoTrace((scene, scene), num_blocks=2), aliases)
        body = RelationalCFG.initial([head], atom("eq", b, a), boolean(True), aliases)
        body.nodes["v0"].region = BlockRegion(None, (Skip(0),))
        loop = LoopRegion(
            atom("eq", b, a),
            (),
            (body.nodes["v0"].region,),
            update=(("b", "c"),),
            invariant=to_z3(pre, ctx),
            body_cfg=body,
            body_demos=((head,),),
        )
        cfg = RelationalCFG.initial([head], pre, boolean(True), aliases)
        cfg.nodes["v0"].region = loop
        result = verified_synthesis(
            cfg,
            lambda n, *_: (n.region, True),
            lambda _: [],
            ctx,
            relations=("equality",),
            variables=1,
            min_blocks=2,
            max_blocks=2,
        )
        self.assertTrue(
            bool(result),
            f"{result.status}: {result.reason}; {result.symbolic}; {result.motion}",
        )
        self.assertIn("invariant_refined", [h["stage"] for h in result.history])
