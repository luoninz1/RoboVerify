import unittest

from synthesis.cfg.kleene import (
    anti_unify,
    carried_bindings,
    encode_fragment,
    match_template,
)
from synthesis.cfg.quotient import find_repetition
from synthesis.predicates.term import atom, ref


class QuotientTests(unittest.TestCase):
    def test_three_iterations_derive_carried_update_and_rebound_source(self):
        labels = [
            atom("ON", ref(a), ref(b))
            for a, b in [("b1", "b0"), ("b2", "b1"), ("b3", "b2")]
        ]
        result = find_repetition(labels)
        self.assertIsNotNone(result)
        self.assertEqual(
            (result.start, result.width, len(result.substitutions)), (0, 1, 3)
        )
        carry, rebound = carried_bindings(result.template)
        self.assertEqual(carry, {"p1": "p0"})
        self.assertEqual(rebound, ("p0",))
        self.assertEqual(result.template.first["p1"], ref("b0"))
        # p1 -> b, p0 -> b_prime yields Assign(b, b_prime).

    def test_template_names_do_not_capture_existing_free_names(self):
        from synthesis.predicates.term import substitute

        for existing in ("p0", "p1", "p2"):
            left = atom("ON", ref("a"), ref(existing))
            right = atom("ON", ref("b"), ref(existing))
            template = anti_unify(encode_fragment([left]), encode_fragment([right]))
            self.assertNotIn(existing, template.first)
            self.assertEqual(
                substitute(template.word[0].predicate, template.first), left
            )
            self.assertEqual(
                substitute(template.word[0].predicate, template.second), right
            )

    def test_repetition_stops_when_carried_update_changes(self):
        labels = [
            atom("ON", ref(a), ref(b))
            for a, b in [("b1", "b0"), ("b2", "b1"), ("b4", "b3")]
        ]
        repeated = find_repetition(labels)
        self.assertEqual(len(repeated.substitutions), 2)

    def test_repetition_rejects_noninjective_later_substitution(self):
        labels = [
            atom("ON", ref(a), ref(b))
            for a, b in [("b1", "b0"), ("b2", "b1"), ("b2", "b2"), ("b3", "b2")]
        ]
        repeated = find_repetition(labels)
        # The third would make both roles carry the same previous value.
        self.assertEqual(len(repeated.substitutions), 2)

    def test_historical_unstack_guard_spike_derives_carried_update(self):
        from synthesis.predicates.term import conjunction, disjunction, forall, negate

        # Lifted from demo_sources.unstack_oracle; no loop supplied to matching.
        labels = []
        for i in range(3):
            bp, b, n = ref(f"b{i+1}"), ref(f"b{i}"), ref("n")
            labels.append(
                conjunction(
                    forall(
                        ["n"],
                        disjunction(atom("eq", bp, n), negate(atom("ON_star", n, bp))),
                    ),
                    negate(atom("eq", bp, b)),
                )
            )
        repeated = find_repetition(labels)
        self.assertEqual((repeated.width, len(repeated.substitutions)), (1, 3))
        carry, rebound = carried_bindings(repeated.template)
        self.assertEqual(len(carry), 1)
        current, next_ = next(iter(carry.items()))
        self.assertEqual(repeated.template.first[current], ref("b0"))
        self.assertEqual(repeated.template.first[next_], ref("b1"))
        self.assertIn(next_, rebound)

    def test_get_bound_arguments_match_by_template_role(self):
        from synthesis.cfg.kleene import Letter

        a = Letter(atom("ON", ref("a"), ref("b")), frozenset(["a"]))
        b = Letter(atom("ON", ref("c"), ref("a")), frozenset(["c"]))
        template = anti_unify((a,), (b,))
        self.assertIsNotNone(template)
        self.assertEqual(template.word[0].get_bound, frozenset(["p0"]))
        wrong = Letter(b.predicate, frozenset(["a"]))
        self.assertIsNone(match_template(template, (wrong,)))

    def test_length_two_repeated_units_are_kept(self):
        a = lambda rel, x, y: atom(rel, ref(x), ref(y))
        labels = [
            a("ON", "b1", "b0"),
            a("Higher", "b1", "b0"),
            a("ON", "b2", "b1"),
            a("Higher", "b2", "b1"),
        ]
        result = find_repetition(labels)
        self.assertEqual(result.width, 2)

    def test_noninjective_first_substitution_is_not_composed(self):
        labels = [atom("ON", ref("a"), ref("a")), atom("ON", ref("b"), ref("c"))]
        template = anti_unify(encode_fragment(labels[:1]), encode_fragment(labels[1:]))
        with self.assertRaises(ValueError):
            carried_bindings(template)


if __name__ == "__main__":
    unittest.main()


class FullQuotientTests(unittest.TestCase):
    def test_full_flat_fold_learns_guard_and_lowers_carried_assignment(self):
        import z3

        from synthesis.api.instructions import Assign, While
        from synthesis.cfg.demos import DemoAssignment, DemoSegment, DemoTrace
        from synthesis.cfg.graph import Edge, Node, RelationalCFG
        from synthesis.cfg.lower import lower
        from synthesis.cfg.quotient import quotient
        from synthesis.cfg.region import BlockRegion
        from synthesis.predicates.scene import Scene
        from synthesis.predicates.term import boolean
        from synthesis.verification_lib.highlevel_verification_lib import (
            HighLevelContext,
        )

        positions = {i: (0, 0, 0.425 + 0.05 * i) for i in range(4)}
        scene = Scene(positions, {f"b{i}": i for i in range(4)})
        trace = DemoTrace((scene, scene, scene, scene))
        names = ["v0", "v1", "v2"]
        labels = [atom("ON", ref(f"b{i+1}"), ref(f"b{i}")) for i in range(3)]
        nodes = {
            name: Node(
                name,
                BlockRegion(
                    (Assign("cursor", f"b{i+1}"),), (Assign("cursor", f"b{i+1}"),)
                ),
            )
            for i, name in enumerate(names)
        }
        edges = [
            Edge("entry", "v0", boolean(True)),
            Edge("v0", "v1", labels[0]),
            Edge("v1", "v2", labels[1]),
            Edge("v2", "exit", labels[2]),
        ]
        demos = DemoAssignment(
            {
                name: [DemoSegment(0, i, i + 1, trace, scene.bindings)]
                for i, name in enumerate(names)
            }
        )
        cfg = RelationalCFG(nodes, edges, names, demos, initial_scope=frozenset(["b0"]))
        from copy import deepcopy

        unresolved = deepcopy(cfg)
        for node in unresolved.nodes.values():
            node.region = None
        self.assertTrue(
            quotient(
                unresolved, infer_invariant=lambda rows, g, scope: z3.BoolVal(True)
            )
        )
        discovered = unresolved.nodes[unresolved.order[0]].region
        self.assertIsNone(discovered.body[0].symbolic)
        self.assertEqual(len(discovered.body_cfg.demos.for_node("body0")), 3)
        self.assertEqual(len(discovered.exit_demos), 1)

        from synthesis.api.instructions import Pick, PickByName

        physical_cfg = deepcopy(cfg)
        for i, name in enumerate(names):
            physical_cfg.nodes[name].region = BlockRegion(None, (Pick(i + 1),))
        self.assertTrue(
            quotient(
                physical_cfg, infer_invariant=lambda rows, g, scope: z3.BoolVal(True)
            )
        )
        physical = lower(physical_cfg, HighLevelContext(), physical=True)
        self.assertIsInstance(physical.instructions[1].body[0], PickByName)
        self.assertEqual(physical.instructions[1].body[0].grab_box_name, "b_prime")
        with self.assertRaisesRegex(ValueError, "relational summary"):
            lower(physical_cfg, HighLevelContext())
        self.assertTrue(
            quotient(cfg, infer_invariant=lambda rows, g, scope: z3.BoolVal(True))
        )
        cfg.validate_structure()
        program = lower(cfg, HighLevelContext())
        self.assertEqual(
            (program.instructions[0].left, program.instructions[0].right), ("b", "b0")
        )
        loop = program.instructions[1]
        self.assertIsInstance(loop, While)
        self.assertEqual((loop.body[-1].left, loop.body[-1].right), ("b", "b_prime"))
        self.assertIsNone(loop.max_iters)
        self.assertEqual(
            [(s.t_start, s.t_end) for s in cfg.demos.for_node(cfg.order[0])], [(0, 3)]
        )
        self.assertEqual(
            [s.entry_index for s in cfg.nodes[cfg.order[0]].region.body_demos[0]],
            [0, 0, 0],
        )
