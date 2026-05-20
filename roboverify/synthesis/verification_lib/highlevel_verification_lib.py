from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from z3 import (
    And,
    BoolSort,
    Const,
    Consts,
    DeclareSort,
    EnumSort,
    Exists,
    ForAll,
    Function,
    If,
    Implies,
    Not,
    Or,
    Solver,
    parse_smt2_string,
    sat,
    unsat,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


@dataclass
class InvariantSpec:
    data: Dict[str, Any]


class HighLevelContext:
    def __init__(
        self,
        mode: str = "declare",
        num_blocks: Optional[int] = None,
        enum_names: Optional[List[str]] = None,
        num_goals: Optional[int] = None,
        goal_enum_names: Optional[List[str]] = None,
        use_tbl: bool = False,
        exists_top: bool = False,
        visualize_enum_scene: bool = False,
        visualization_prefix: str = "highlevel_scene",
        verification_mode: str = "box",
    ):
        self.mode = mode
        self.num_blocks = num_blocks
        self.enum_names = enum_names
        self.num_goals = num_goals
        self.goal_enum_names = goal_enum_names
        self.use_tbl = use_tbl
        self.exists_top = exists_top
        self.visualize_enum_scene = visualize_enum_scene
        self.visualization_prefix = visualization_prefix
        self.verification_mode = verification_mode
        self.enum_blocks: List[Any] = []
        self.enum_goals: List[Any] = []
        self.goal_enum_names_effective: List[str] = []
        self._build_symbols()
        self._build_goal_symbols()

    def _build_symbols(self):
        if self.verification_mode == "goals":
            return
        if self.mode == "declare":
            self.BoxSort = DeclareSort("Box")
            self.enum_blocks = []
            self.enum_names_effective: List[str] = []
        elif self.mode == "enum":
            names = self.enum_names
            if names is None:
                if self.num_blocks is None:
                    raise ValueError("enum mode requires enum_names or num_blocks.")
                names = [f"b{9+i}" for i in range(self.num_blocks)]
            self.BoxSort, enum_consts = EnumSort("Box", names)
            self.enum_blocks = list(enum_consts)
            self.enum_names_effective = list(names)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.ON_star = Function("ON_star", self.BoxSort, self.BoxSort, BoolSort())
        self.ON_star_zero = Function(
            "ON_star_zero", self.BoxSort, self.BoxSort, BoolSort()
        )
        self.Higher = Function("Higher", self.BoxSort, self.BoxSort, BoolSort())
        self.Scattered = Function("Scattered", self.BoxSort, self.BoxSort, BoolSort())
        self.Top = Function("Top", self.BoxSort, BoolSort())

    def _resolve_goal_enum_names(self) -> List[str]:
        """Finite Goal universe for ``mode=='enum'`` (goal nodes only, no ``null`` ctor)."""
        if self.goal_enum_names is not None:
            names = list(self.goal_enum_names)
            if not names:
                raise ValueError("goal_enum_names must be a non-empty list.")
            if "null" in names:
                raise ValueError(
                    'goal_enum_names must not include "null"; null is a separate Goal constant.'
                )
            return names
        if self.num_goals is not None:
            return [f"g{i}" for i in range(self.num_goals)]
        raise ValueError(
            "mode='enum' with verification_mode='goals' requires goal_enum_names or "
            "num_goals to define the finite Goal universe."
        )

    def _build_goal_symbols(self):
        self.GoalSort = None
        self.null = None
        self.d_star = None
        self.r_star = None
        self.l0 = None
        self.Mark = None
        self.enum_goals = []
        self.goal_enum_names_effective = []
        if self.verification_mode != "goals":
            return
        if self.mode == "declare":
            self.GoalSort = DeclareSort("Goal")
            self.null = Const("null", self.GoalSort)
        elif self.mode == "enum":
            gnames = self._resolve_goal_enum_names()
            self.GoalSort, goal_consts = EnumSort("Goal", gnames)
            self.enum_goals = list(goal_consts)
            self.goal_enum_names_effective = list(gnames)
            self.null = Const("null", self.GoalSort)
        else:
            raise ValueError(f"Unknown mode for GoalSort: {self.mode}")
        self.d_star = Function("d_star", self.GoalSort, self.GoalSort, BoolSort())
        self.r_star = Function("r_star", self.GoalSort, self.GoalSort, BoolSort())
        self.l0 = Function("l0", self.GoalSort, self.GoalSort)
        self.Mark = Function("Mark", self.GoalSort, BoolSort())

    def get_consts(self, symbol: str):
        (c,) = Consts(symbol, self.BoxSort)
        return c

    def get_goal_consts(self, symbol: str):
        if self.GoalSort is None:
            raise ValueError(
                "GoalSort is not configured. Initialize HighLevelContext with "
                "verification_mode='goals'."
            )
        (c,) = Consts(symbol, self.GoalSort)
        return c

    def add_axiom_higher(self, s: Solver):
        x, y, c = Consts("x y c", self.BoxSort)
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(And(self.Higher(x, y), self.Higher(y, c)), self.Higher(x, c)),
            ),
            "higher1",
        )
        s.assert_and_track(ForAll([x], self.Higher(x, x)), "higher2")
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(self.Higher(x, y), self.Higher(x, c)),
                    Or(self.Higher(y, c), self.Higher(c, y)),
                ),
            ),
            "higher3",
        )
        if self.use_tbl:
            tbl = self.get_consts("tbl")
            s.assert_and_track(
                ForAll(
                    [x],
                    Implies(Or(self.Higher(x, tbl), self.Higher(tbl, x)), x == tbl),
                ),
                "higher_tbl",
            )

    def add_axiom_scattered(self, s: Solver):
        x, y, c = Consts("x y c", self.BoxSort)
        s.assert_and_track(
            ForAll([x, y], self.Scattered(x, y) == self.Scattered(y, x)), "scattered1"
        )
        s.assert_and_track(ForAll([x], Not(self.Scattered(x, x))), "scattered2")
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    self.ON_star(x, y),
                    self.Scattered(x, c) == self.Scattered(y, c),
                ),
            ),
            "scattered3",
        )

    def add_axiom_goal_nested(self, s: Solver):
        if self.GoalSort is None:
            return

        x, y, z = Consts("x y z", self.GoalSort)
        # dtca on d_star
        s.assert_and_track(ForAll([x], self.d_star(x, x)), "d_refl")
        s.assert_and_track(
            ForAll(
                [x, y, z],
                Implies(
                    And(self.d_star(x, y), self.d_star(y, z)),
                    self.d_star(x, z),
                ),
            ),
            "d_trans",
        )
        s.assert_and_track(
            ForAll(
                [x, y, z],
                Implies(
                    And(self.d_star(x, y), self.d_star(x, z)),
                    Or(self.d_star(y, z), self.d_star(z, y)),
                ),
            ),
            "d_lin",
        )
        s.assert_and_track(
            ForAll(
                [x, y],
                Implies(
                    And(self.d_star(x, y), self.d_star(y, x)),
                    x == y,
                ),
            ),
            "d_antisymm",
        )
        s.assert_and_track(
            ForAll(
                [x],
                Implies(
                    Or(self.d_star(self.null, x), self.d_star(x, self.null)),
                    self.null == x,
                ),
            ),
            "d_refl_isolated_null",
        )

        # dtca on r_star
        s.assert_and_track(ForAll([x], self.r_star(x, x)), "r_refl")
        s.assert_and_track(
            ForAll(
                [x, y, z],
                Implies(
                    And(self.r_star(x, y), self.r_star(y, z)),
                    self.r_star(x, z),
                ),
            ),
            "r_trans",
        )
        s.assert_and_track(
            ForAll(
                [x, y, z],
                Implies(
                    And(self.r_star(x, y), self.r_star(x, z)),
                    Or(self.r_star(y, z), self.r_star(z, y)),
                ),
            ),
            "r_lin",
        )
        s.assert_and_track(
            ForAll(
                [x, y],
                Implies(
                    And(self.r_star(x, y), self.r_star(y, x)),
                    x == y,
                ),
            ),
            "r_antisymm",
        )
        s.assert_and_track(
            ForAll(
                [x],
                Implies(
                    Or(self.r_star(self.null, x), self.r_star(x, self.null)),
                    self.null == x,
                ),
            ),
            "r_refl_isolated_null",
        )

        # l0 + interaction axioms
        s.assert_and_track(ForAll([x], self.r_star(self.l0(x), x)), "l0_1")
        s.assert_and_track(
            ForAll([x, y], Implies(self.r_star(x, y), self.r_star(self.l0(y), x))),
            "l0_2",
        )
        s.assert_and_track(
            ForAll(
                [x, y, z],
                Implies(
                    And(self.r_star(x, y), self.d_star(z, y)),
                    Or(x == y, z == y),
                ),
            ),
            "nested_1",
        )
        s.assert_and_track(
            ForAll(
                [x, y],
                Implies(And(self.d_star(x, y), x != y), self.l0(x) == x),
            ),
            "nested_2",
        )

        # global reachability axiom
        h = Const("h", self.GoalSort)
        s.assert_and_track(
            ForAll([x], Implies(x != self.null, self._dr_reach(h, x))),
            "dr_reach",
        )

    def _f_plus(self, rel, a, b):
        return And(rel(a, b), a != b)

    def f_(self, rel, a, b):
        if self.GoalSort is None:
            raise ValueError(
                "Goal relational helpers require verification_mode='goals'."
            )
        t = Const("t_f", self.GoalSort)
        return And(
            self._f_plus(rel, a, b),
            ForAll([t], Implies(self._f_plus(rel, a, t), rel(b, t))),
        )

    def f_tot(self, rel, a, b):
        if self.GoalSort is None:
            raise ValueError(
                "Goal relational helpers require verification_mode='goals'."
            )
        t = Const("t_ft", self.GoalSort)
        return Or(
            self.f_(rel, a, b),
            And(b == self.null, ForAll([t], Not(self._f_plus(rel, a, t)))),
        )

    def dtot(self, a, b):
        return self.f_tot(self.d_star, a, b)

    def rtot(self, a, b):
        return self.f_tot(self.r_star, a, b)

    def _dr_reach(self, x, y):
        if self.GoalSort is None:
            raise ValueError(
                "Goal relational helpers require verification_mode='goals'."
            )
        return self.d_star(x, self.l0(y))

    def _flat_order(self, a, b):
        if self.GoalSort is None:
            raise ValueError(
                "Goal relational helpers require verification_mode='goals'."
            )
        return If(
            self.l0(a) == self.l0(b),
            self.r_star(a, b),
            self.d_star(self.l0(a), self.l0(b)),
        )

    def _flat_between(self, a, b, c):
        return And(self._flat_order(a, b), self._flat_order(b, c))

    def add_axiom(self, s: Solver):
        x, y, c = Consts("x y c", self.BoxSort)
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(self.ON_star(x, y), self.ON_star(y, c)),
                    self.ON_star(x, c),
                ),
            ),
            "on1",
        )
        s.assert_and_track(ForAll([x], self.ON_star(x, x)), "on2")
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(self.ON_star(x, y), self.ON_star(x, c)),
                    Or(self.ON_star(y, c), self.ON_star(c, y)),
                ),
            ),
            "on3",
        )
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(self.ON_star(x, c), self.ON_star(y, c)),
                    Or(self.ON_star(x, y), self.ON_star(y, x)),
                ),
            ),
            "on4",
        )
        s.assert_and_track(
            ForAll(
                [x, y],
                Implies(self.ON_star(x, y), Implies(self.ON_star(y, x), x == y)),
            ),
            "on5",
        )
        if self.use_tbl:
            tbl = self.get_consts("tbl")
            s.assert_and_track(
                ForAll(
                    [x],
                    Implies(Or(self.ON_star(x, tbl), self.ON_star(tbl, x)), x == tbl),
                ),
                "on_tbl",
            )
        if self.exists_top:
            s.assert_and_track(
                ForAll(
                    [x],
                    Exists(
                        [y],
                        And(
                            ForAll([c], Implies(self.ON_star(c, y), c == y)),
                            self.ON_star(y, x),
                        ),
                    ),
                ),
                "on_exists_top",
            )

    def add_axiom_on_star_zero(self, s: Solver):
        x, y, c = Consts("x y c", self.BoxSort)
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(self.ON_star_zero(x, y), self.ON_star_zero(y, c)),
                    self.ON_star_zero(x, c),
                ),
            ),
            "on_zero_1",
        )
        s.assert_and_track(ForAll([x], self.ON_star_zero(x, x)), "on_zero_2")
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(self.ON_star_zero(x, y), self.ON_star_zero(x, c)),
                    Or(self.ON_star_zero(y, c), self.ON_star_zero(c, y)),
                ),
            ),
            "on_zero_3",
        )
        s.assert_and_track(
            ForAll(
                [x, y, c],
                Implies(
                    And(self.ON_star_zero(x, c), self.ON_star_zero(y, c)),
                    Or(self.ON_star_zero(x, y), self.ON_star_zero(y, x)),
                ),
            ),
            "on_zero_4",
        )
        s.assert_and_track(
            ForAll(
                [x, y],
                Implies(
                    self.ON_star_zero(x, y), Implies(self.ON_star_zero(y, x), x == y)
                ),
            ),
            "on_zero_5",
        )
        if self.use_tbl:
            tbl = self.get_consts("tbl")
            s.assert_and_track(
                ForAll(
                    [x],
                    Implies(
                        Or(self.ON_star_zero(x, tbl), self.ON_star_zero(tbl, x)),
                        x == tbl,
                    ),
                ),
                "on_zero_tbl",
            )
        if self.exists_top:
            s.assert_and_track(
                ForAll(
                    [x],
                    Exists(
                        [y],
                        And(
                            ForAll([c], Implies(self.ON_star_zero(c, y), c == y)),
                            self.ON_star_zero(y, x),
                        ),
                    ),
                ),
                "on_zero_exists_top",
            )

    def _print_relation_table(self, model, relation, relation_name: str):
        if not self.enum_blocks:
            return
        print(f"\n{relation_name} table:")
        print("      " + " ".join(f"{n:>6}" for n in self.enum_names_effective))
        for name_i, box_i in zip(self.enum_names_effective, self.enum_blocks):
            row = [f"{name_i:>6}"]
            for box_j in self.enum_blocks:
                val = model.evaluate(relation(box_i, box_j), model_completion=True)
                row.append(f"{str(val):>6}")
            print(" ".join(row))

    def _extract_direct_on(self, model, relation) -> Dict[str, str]:
        direct_on = {n: "table" for n in self.enum_names_effective}
        for a_name, a in zip(self.enum_names_effective, self.enum_blocks):
            for b_name, b in zip(self.enum_names_effective, self.enum_blocks):
                if a_name == b_name:
                    continue
                if not model.evaluate(relation(a, b), model_completion=True):
                    continue
                has_middle = False
                for c_name, c in zip(self.enum_names_effective, self.enum_blocks):
                    if c_name in [a_name, b_name]:
                        continue
                    if model.evaluate(
                        relation(a, c), model_completion=True
                    ) and model.evaluate(relation(c, b), model_completion=True):
                        has_middle = True
                        break
                if not has_middle:
                    direct_on[a_name] = b_name
        return direct_on

    def _build_stacks(self, direct_on: Dict[str, str]) -> List[List[str]]:
        stacks = []
        visited = set()
        bases = [b for b, v in direct_on.items() if v == "table"]
        for base in bases:
            if base in visited:
                continue
            stack = [base]
            visited.add(base)
            top = base
            while True:
                above = None
                for b, v in direct_on.items():
                    if v == top and b not in visited:
                        above = b
                        break
                if above is None:
                    break
                stack.append(above)
                visited.add(above)
                top = above
            stacks.append(stack)
        return stacks

    def _draw_stacks(self, stacks: List[List[str]], filename: str, title: str) -> None:
        if Image is None:
            return
        img = Image.new("RGB", (800, 400), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), title, fill="black", font=font)
        block_w = 70
        block_h = 30
        gap = 60
        x = 50
        for stack in stacks:
            y = 350
            for block in stack:
                draw.rectangle(
                    [x, y - block_h, x + block_w, y], outline="black", width=2
                )
                draw.text((x + 15, y - block_h + 5), block, fill="black", font=font)
                y -= block_h + 5
            draw.line([x, y + 5, x + block_w, y + 5], fill="black", width=3)
            x += block_w + gap
        img.save(filename)

    def _visualize_enum(self, model, viz_tag: Optional[str] = None) -> None:
        self._print_relation_table(model, self.ON_star, "ON_star")
        self._print_relation_table(model, self.ON_star_zero, "ON_star_zero")
        self._print_relation_table(model, self.Higher, "Higher")
        self._print_relation_table(model, self.Scattered, "Scattered")
        if self.visualize_enum_scene:
            prefix = self.visualization_prefix
            if viz_tag:
                prefix = f"{prefix}_{viz_tag}"
            start_state = self._extract_direct_on(model, self.ON_star_zero)
            current_state = self._extract_direct_on(model, self.ON_star)
            self._draw_stacks(
                self._build_stacks(start_state),
                f"{prefix}_start_state.png",
                "Start State",
            )
            self._draw_stacks(
                self._build_stacks(current_state),
                f"{prefix}_current_state.png",
                "Current State",
            )

    def start_verification(self, vc=None, viz_tag: Optional[str] = None) -> None:
        s = Solver()
        self.add_axiom(s)
        self.add_axiom_on_star_zero(s)
        self.add_axiom_higher(s)
        self.add_axiom_scattered(s)
        self.add_axiom_goal_nested(s)
        if vc is not None:
            s.add(Not(vc))
        result = s.check()
        if result == sat:
            print("VC is satisfiable")
            model = s.model()
            print(model)
            if self.mode == "enum":
                self._visualize_enum(model, viz_tag=viz_tag)
        elif result == unsat:
            print("VC is unsatisfiable")
        else:
            print(result)

    def check_satisfiable(
        self,
        formula=None,
        visualize_model: bool = True,
        viz_tag: Optional[str] = None,
    ):
        """Check satisfiability under axioms without negating formula.

        Returns (z3_result, model_or_none). The model is present only when result is sat.
        """
        s = Solver()
        if self.verification_mode == "goals":
            self.add_axiom_goal_nested(s)
        else:
            self.add_axiom(s)
            self.add_axiom_on_star_zero(s)
            self.add_axiom_higher(s)
            self.add_axiom_scattered(s)

        if formula is not None:
            s.add(formula)
        result = s.check()
        model = None
        if result == sat:
            print("satisfiable")
            model = s.model()
            print(model)
            if (
                visualize_model
                and self.mode == "enum"
                and self.verification_mode != "goals"
            ):
                self._visualize_enum(model, viz_tag=viz_tag)
        elif result == unsat:
            print("unsatisfiable")
        else:
            print(result)
        return result, model

    def check_satisfiable_with_pq(
        self,
        p=None,
        q=None,
        visualize_model: bool = True,
        viz_tag: Optional[str] = None,
    ):
        """Check satisfiability under axioms without negating formula.

        Returns (z3_result, model_or_none). The model is present only when result is sat.
        """
        s = Solver()
        if self.verification_mode == "goals":
            self.add_axiom_goal_nested(s)
        else:
            self.add_axiom(s)
            self.add_axiom_on_star_zero(s)
            self.add_axiom_higher(s)
            self.add_axiom_scattered(s)

        if p is not None:
            s.add(p)
        if q is not None:
            s.add(Not(q))
        result = s.check()
        model = None
        if result == sat:
            print("satisfiable")
            model = s.model()
            print(model)
            if (
                visualize_model
                and self.mode == "enum"
                and self.verification_mode != "goals"
            ):
                self._visualize_enum(model, viz_tag=viz_tag)
        elif result == unsat:
            print("unsatisfiable")
        else:
            print(result)
        return result, model

    def expr_to_spec(self, expr) -> InvariantSpec:
        return InvariantSpec(data={"sexpr": expr.sexpr()})

    def spec_to_expr(
        self, spec: InvariantSpec, known_const_names: Optional[List[str]] = None
    ):
        sexpr = spec.data["sexpr"]
        decls: Dict[str, Any] = {
            "ON_star": self.ON_star,
            "ON_star_zero": self.ON_star_zero,
            "Higher": self.Higher,
            "Scattered": self.Scattered,
            "Top": self.Top,
        }
        for name in known_const_names or []:
            decls[name] = Const(name, self.BoxSort)
        parsed = parse_smt2_string(
            f"(assert {sexpr})", sorts={"Box": self.BoxSort}, decls=decls
        )
        return parsed[0]


Variable_pools = {}


class highlevel_z3_solver:
    def __init__(
        self,
        use_tbl: bool = False,
        box_sort_mode: str = "declare",
        num_blocks: Optional[int] = None,
        enum_names: Optional[List[str]] = None,
        num_goals: Optional[int] = None,
        goal_enum_names: Optional[List[str]] = None,
        visualize_enum_scene: bool = False,
        visualization_prefix: str = "highlevel_scene",
        verification_mode: str = "box",
    ):
        self.context = HighLevelContext(
            mode=box_sort_mode,
            num_blocks=num_blocks,
            enum_names=enum_names,
            num_goals=num_goals,
            goal_enum_names=goal_enum_names,
            use_tbl=use_tbl,
            visualize_enum_scene=visualize_enum_scene,
            visualization_prefix=visualization_prefix,
            verification_mode=verification_mode,
        )

    def start_verification(self, vc=None, viz_tag: Optional[str] = None) -> None:
        return self.context.start_verification(vc, viz_tag=viz_tag)

    def check_satisfiable(
        self,
        formula=None,
        visualize_model: bool = False,
        viz_tag: Optional[str] = None,
    ):
        return self.context.check_satisfiable(
            formula,
            visualize_model=visualize_model,
            viz_tag=viz_tag,
        )

    def add_axiom(self, s: Solver):
        return self.context.add_axiom(s)

    def add_axiom_on_star_zero(self, s: Solver):
        return self.context.add_axiom_on_star_zero(s)

    def add_axiom_higher(self, s: Solver):
        return self.context.add_axiom_higher(s)

    def add_axiom_scattered(self, s: Solver):
        return self.context.add_axiom_scattered(s)

    def add_axiom_goal_nested(self, s: Solver):
        return self.context.add_axiom_goal_nested(s)
