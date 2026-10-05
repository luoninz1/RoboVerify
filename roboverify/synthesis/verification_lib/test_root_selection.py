import unittest
from unittest.mock import patch

import z3

from synthesis.api.instructions import Assign
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.root_selection import RootContext


class RootSelectionTests(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext()
        self.top, self.root, self.other = [
            self.context.get_consts(n) for n in ("top", "z_base", "b0")
        ]
        u = z3.FreshConst(self.context.BoxSort)
        self.entry = z3.And(
            self.context.ON_star(self.top, self.root),
            self.top != self.root,
            self.other != self.root,
            z3.ForAll(
                [u], z3.Implies(self.context.ON_star(self.root, u), u == self.root)
            ),
        )

    def test_root_is_proved_not_chosen_by_name(self):
        result = RootContext(self.context, self.entry).select(
            "top", ["b0", "top", "z_base"]
        )
        self.assertEqual((result.status, result.root), ("valid", "z_base"))

    def test_named_chain_does_not_exclude_unnamed_lower_objects(self):
        entry = z3.And(self.context.ON_star(self.top, self.root), self.top != self.root)
        result = RootContext(self.context, entry).select("top", ["top", "z_base"])
        self.assertEqual(result.status, "unproved")
        self.assertIsNone(result.root)

    def test_unknown_and_inconsistent_contexts_have_no_root(self):
        result = RootContext(self.context, z3.BoolVal(False)).select("top", ["z_base"])
        self.assertEqual(result.status, "inconsistent")
        self.assertIsNone(result.root)
        with patch("z3.Solver.check", return_value=z3.unknown):
            result = RootContext(self.context, self.entry).select("top", ["z_base"])
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.root)

    def test_unavailable_root_has_no_fallback(self):
        result = RootContext(self.context, self.entry).select("top", ["top", "b0"])
        self.assertEqual(result.status, "unproved")
        self.assertIsNone(result.root)

    def test_wp_and_assignment_history_use_current_bindings(self):
        roots = RootContext(self.context, self.entry)
        roots.history.append(Assign("cursor", "top"))
        output = self.context.get_consts("output")
        result = roots.select(
            "cursor",
            ["z_base", "cursor"],
            continuation=[Assign("output", "cursor")],
            postcondition=self.context.ON_star(output, self.root),
        )
        self.assertEqual((result.status, result.root), ("valid", "z_base"))
        bad = roots.select(
            "cursor",
            ["z_base"],
            continuation=[Assign("output", "cursor")],
            postcondition=z3.BoolVal(False),
        )
        self.assertEqual(bad.status, "unproved")
        self.assertIsNone(bad.root)

    def test_table_marker_is_not_a_root_candidate(self):
        result = RootContext(self.context, self.entry).select("top", ["tbl", "sym"])
        self.assertEqual(result.status, "unproved")
