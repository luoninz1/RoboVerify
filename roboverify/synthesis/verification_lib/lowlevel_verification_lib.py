from itertools import product
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from z3 import (
    Z3_OP_DISTINCT,
    Z3_OP_EQ,
    Z3_OP_IMPLIES,
    Z3_OP_ITE,
    Z3_OP_NOT,
    Z3_OP_UNINTERPRETED,
    Z3_OP_XOR,
    Abs,
    And,
    BoolVal,
    Consts,
    DeclareSort,
    Exists,
    ForAll,
    Function,
    If,
    Not,
    Or,
    Real,
    Reals,
    RealSort,
    RealVal,
    Solver,
    get_var_index,
    is_app,
    is_bool,
    is_quantifier,
    is_true,
    is_var,
    sat,
    unsat,
)

import synthesis.api.instructions as instructions
from synthesis.util import on
from synthesis.util.symbols import fresh_const, open_quantifier


class UnsupportedMotionInstruction(Exception):
    """An instruction reached motion verification that it cannot classify.

    Raised rather than skipped: silently passing over an unrecognised
    instruction is what let a body be reported as verified when part of it was
    never examined.
    """


# Carry no geometry, so there is genuinely nothing for the motion level to check.
# ``Assign`` is the common case: the lowered loop bodies append the loop-carried
# update (see ``verify_stack_with_learned_invariant``), so this branch is taken on
# every run.
INERT_MOTION_INSTRUCTIONS = (instructions.Assign, instructions.Skip)

# Do move the end effector through space, but have no collision encoding here yet.
# They must fail closed until one exists; passing over them was unsound.
UNHANDLED_MOTION_INSTRUCTIONS = (
    instructions.Pick,
    instructions.PickByName,
    instructions.Move,
    instructions.MoveByName,
    instructions.Release,
    instructions.ReleaseByName,
    instructions.PickPlace,
)


#: Name of the Box-sort element standing for the table. The relational level
#: gives it no geometry -- the axioms isolate it from all three relations -- so
#: the predicates below test for it by identity instead of by coordinates.
TABLE_CONST_NAME = "tbl"


def abs_diff(u, v):
    return If(u - v >= 0, u - v, v - u)


def draw_cube(ax, cx, cy, cz, L, color="blue"):
    """Draw a cube centered at (cx, cy, cz) with side length L."""
    r = L / 2.0
    v = [
        (cx - r, cy - r, cz - r),
        (cx + r, cy - r, cz - r),
        (cx + r, cy + r, cz - r),
        (cx - r, cy + r, cz - r),
        (cx - r, cy - r, cz + r),
        (cx + r, cy - r, cz + r),
        (cx + r, cy + r, cz + r),
        (cx - r, cy + r, cz + r),
    ]
    faces = [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[7], v[3], v[0]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=0.3))


def _box_corners(center, halfwidth):
    cx, cy, cz = center
    h = float(halfwidth)
    return np.array(
        [
            [cx - h, cy - h, cz - h],
            [cx - h, cy - h, cz + h],
            [cx - h, cy + h, cz - h],
            [cx - h, cy + h, cz + h],
            [cx + h, cy - h, cz - h],
            [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz - h],
            [cx + h, cy + h, cz + h],
        ],
        dtype=float,
    )


def draw_encoded_tube(ax, p0, p1, halfwidth, color="purple", alpha=0.12):
    """Visualize the swept AABB 'tube' used by `encode_collision` as ONE object.

    The swept volume is the convex hull of the two endpoint AABBs centered at p0/p1
    with halfwidth `halfwidth` (since the constraint is Abs(coord - c(t)) < halfwidth).
    """
    x0, y0, z0 = p0
    x1, y1, z1 = p1

    # Draw the centerline.
    ax.plot([x0, x1], [y0, y1], [z0, z1], color=color, linewidth=2, alpha=0.6)

    pts = np.vstack([_box_corners(p0, halfwidth), _box_corners(p1, halfwidth)])
    hull = ConvexHull(pts)

    faces = []
    for simplex in hull.simplices:
        tri = pts[simplex]
        faces.append(tri.tolist())

    poly = Poly3DCollection(faces, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(poly)


class LowLevelContext:
    def __init__(
        self,
        sort_name: str = "Box",
        default_L: float = 1.0,
        use_tbl: bool = False,
        higher_tolerance: float | None = None,
    ):
        self.sort_name = sort_name
        self.default_L = default_L
        self.higher_tolerance = (
            on.get_higher_tolerance()
            if higher_tolerance is None
            else on.validate_higher_tolerance(higher_tolerance)
        )
        # Mirrors HighLevelContext.use_tbl: the tbl axioms are only asserted when
        # the task actually has a table, and the relations below are only
        # narrowed when those axioms are in force. Turning it on for a task with
        # no table would add an unconstrained sort element that weakens every
        # condition for nothing.
        self.use_tbl = use_tbl
        self._build_symbols()

    def _build_symbols(self):
        self.BoxSort = DeclareSort(self.sort_name)
        self.X = Function("X", self.BoxSort, RealSort())
        self.Y = Function("Y", self.BoxSort, RealSort())
        self.Z = Function("Z", self.BoxSort, RealSort())
        # Frozen loop-entry geometry is independent of the current loop head.
        self.X0 = Function("X0", self.BoxSort, RealSort())
        self.Y0 = Function("Y0", self.BoxSort, RealSort())
        self.Z0 = Function("Z0", self.BoxSort, RealSort())
        (self.L,) = Reals("L")

    def lowlevel_box_equal(self, b1, b2):
        # The table has no position. Equal coordinates must never identify a
        # physical block with tbl; real blocks retain geometric equality.
        return self._isolate_table(
            And(
                self.X(b1) == self.X(b2),
                self.Y(b1) == self.Y(b2),
                self.Z(b1) == self.Z(b2),
            ),
            b1,
            b2,
            reflexive=True,
        )

    def table_const(self):
        """The ``tbl`` element of the Box sort.

        ``Consts`` interns by name and sort, so this is the same term the
        high-level ``tbl`` axioms and the translated conditions refer to.
        """
        return self.get_consts(TABLE_CONST_NAME)

    def _is_table(self, b):
        """Whether the box term *b* denotes the table.

        The Box sort is a ``DeclareSort`` with no unique-names assumption, so
        this is a real constraint rather than a syntactic test: two differently
        named constants may denote the same element unless something rules it
        out. When the task has no table, there is no such element and the
        question does not arise.
        """
        if not self.use_tbl:
            return BoolVal(False)
        return b == self.table_const()

    def _isolate_table(self, geometry, b1, b2, reflexive: bool):
        """Wrap a geometric relation so the table is isolated from it.

        The high-level axioms (``on_tbl``, ``higher_tbl``, ``scattered_not_tbl``)
        make every pair involving ``tbl`` false, except that ``ON*`` and
        ``Higher`` are reflexive and so hold at ``(tbl, tbl)``. Without this the
        table's ``X``/``Y``/``Z`` are unconstrained reals the solver may place
        anywhere, so a translated condition could claim a block is on the table,
        or that the table is above a block, purely by choosing coordinates.

        The isolation has to live in the relation rather than in an added axiom:
        asserting ``higher_tbl`` on top of a purely geometric ``Higher`` forces
        ``Z(x) < Z(tbl)`` and ``Z(tbl) < Z(x)`` at once, which makes every
        condition set unsatisfiable.
        """
        if not self.use_tbl:
            return geometry
        neither_is_table = And(
            Not(self._is_table(b1)),
            Not(self._is_table(b2)),
        )
        if not reflexive:
            return And(neither_is_table, geometry)
        return Or(
            And(self._is_table(b1), self._is_table(b2)),
            And(neither_is_table, geometry),
        )

    def lowlevel_on_star(self, b1, b2):
        """Geometric reading of ``ON*``, mirroring ``on.on_star_implementation``."""
        return self._isolate_table(
            And(
                Abs(self.X(b1) - self.X(b2)) < self.L / 2,
                Abs(self.Y(b1) - self.Y(b2)) < self.L / 2,
                self.Z(b1) >= self.Z(b2),
            ),
            b1,
            b2,
            reflexive=True,
        )

    def lowlevel_on_star_zero(self, b1, b2):
        return self._isolate_table(
            And(
                Abs(self.X0(b1) - self.X0(b2)) < self.L / 2,
                Abs(self.Y0(b1) - self.Y0(b2)) < self.L / 2,
                self.Z0(b1) >= self.Z0(b2),
            ),
            b1,
            b2,
            reflexive=True,
        )

    def lowlevel_on_direct(self, b1, b2):
        """Geometric reading of *direct* ``on``, mirroring :func:`on.z3_on`.

        Same xy tolerance as :meth:`lowlevel_on_star` but a bounded vertical gap
        ``0 <= dz < 1.5 L``, so it says *b1 rests on b2* rather than *b1 is
        somewhere up that stack*. Phase D's contract-realization obligation --
        that a synthesized body actually establishes ``on(b', b)`` -- cannot be
        stated without it.

        A block resting on the table is not expressible here and is not meant to
        be: that is a fact about the table's surface height, which belongs to the
        motion level. ``on_direct`` is contained in ``ON*``, which ``on_tbl``
        already makes false for the table, so it is not reflexive at the table
        either.
        """
        return self._isolate_table(
            And(
                Abs(self.X(b1) - self.X(b2)) < self.L / 2,
                Abs(self.Y(b1) - self.Y(b2)) < self.L / 2,
                self.Z(b1) - self.Z(b2) >= 0,
                self.Z(b1) - self.Z(b2) < RealVal(1.5) * self.L,
            ),
            b1,
            b2,
            reflexive=False,
        )

    def lowlevel_higher(self, b1, b2):
        """b1 is at least as high as b2; mirrors ``on.higher_implementation``."""
        return self._isolate_table(
            on.higher_z3(self.Z(b1), self.Z(b2), tolerance=self.higher_tolerance),
            b1,
            b2,
            reflexive=True,
        )

    def lowlevel_scattered(self, t1, t2):
        """Mirrors ``on.scattered_implementation``.

        ``scattered_not_tbl`` plus symmetry and irreflexivity make every pair
        involving the table false, with no reflexive case.

        The separation test is non-strict, matching both
        ``on.scattered_implementation`` and the rejection sampler in
        ``fpp_construction_env._reset_sim_roboverify_stack``, which accepts a
        layout as soon as ``dx >= 2L or dy >= 2L``. It was strict here, so a
        layout the environment can actually produce -- separated by exactly
        ``2L`` -- counted as scattered for the invariant that was learned and
        not scattered for the checker that had to discharge it.
        """
        return self._isolate_table(
            Or(
                Abs(self.X(t1) - self.X(t2)) >= 2 * self.L,
                Abs(self.Y(t1) - self.Y(t2)) >= 2 * self.L,
            ),
            t1,
            t2,
            reflexive=False,
        )

    def get_consts(self, symbol: str):
        (c,) = Consts(symbol, self.BoxSort)
        return c

    def encode_collision(self, a, p0, p1):
        """Return a Z3 formula asserting block *a* collides with the swept
        volume of a cube moving linearly from *p0* to *p1*.

        p0, p1 are (x, y, z) triples of Z3 Real expressions.
        """
        return self.encode_collision_at((self.X(a), self.Y(a), self.Z(a)), p0, p1)

    def encode_collision_at(self, position, p0, p1):
        """The same swept-cube predicate for a block in a later symbolic state."""
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        t = fresh_const(RealSort(), prefix="tube_t")
        cx = (1 - t) * x0 + t * x1
        cy = (1 - t) * y0 + t * y1
        cz = (1 - t) * z0 + t * z1
        return And(
            t >= 0,
            t <= 1,
            Abs(position[0] - cx) < self.L,
            Abs(position[1] - cy) < self.L,
            Abs(position[2] - cz) < self.L,
        )

    def check_solver(
        self, s, blocks, save_path=None, encoded_tube=None, extra_blocks=None
    ):
        """Check satisfiability; visualize the scene if SAT, print core if UNSAT."""
        result = s.check()
        print(f"Satisfiability: {result}")
        if result == unsat:
            print("UNSAT - constraints are unsatisfiable")
            core = s.unsat_core()
            if core:
                print("Unsat core:", core)
        elif result == sat:
            m = s.model()
            print(m)
            self.visualize_scene(
                m,
                blocks,
                save_path=save_path,
                encoded_tube=encoded_tube,
                extra_blocks=extra_blocks,
            )
        return result

    def visualize_scene(
        self, model, blocks, save_path=None, encoded_tube=None, extra_blocks=None
    ):
        """Plot each block as a 3-D cube using coordinates from the Z3 model."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        colors = ["red", "blue", "yellow", "green", "gray", "cyan", "magenta", "orange"]

        def to_float(z3val):
            s = str(model.eval(z3val).as_decimal(20))
            return float(s) if s[-1] != "?" else float(s[:-1])

        L_val = to_float(self.L)
        half_side = L_val / 2.0

        # Track bounds so we can enforce equal aspect ratio.
        min_x = float("inf")
        min_y = float("inf")
        min_z = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        max_z = float("-inf")

        def include_aabb(cx, cy, cz, hx, hy, hz):
            nonlocal min_x, min_y, min_z, max_x, max_y, max_z
            min_x = min(min_x, cx - hx)
            max_x = max(max_x, cx + hx)
            min_y = min(min_y, cy - hy)
            max_y = max(max_y, cy + hy)
            min_z = min(min_z, cz - hz)
            max_z = max(max_z, cz + hz)

        def print_relation_tables(objects, labels):
            def eval_bool(expr):
                return bool(is_true(model.eval(expr, model_completion=True)))

            def print_table(title, rel_fn):
                width = max(5, max(len(s) for s in labels) + 1)
                print("\n" + title)
                print("".ljust(width) + "".join(s.ljust(width) for s in labels))
                for i, (oi, li) in enumerate(zip(objects, labels)):
                    row = [li.ljust(width)]
                    for j, oj in enumerate(objects):
                        cell = "T" if eval_bool(rel_fn(oi, oj)) else "F"
                        row.append(cell.ljust(width))
                    print("".join(row))

            # Display current relations; ON_star_zero has separate frozen coordinates.
            print_table(
                "Relation: ON_star (lowlevel_on_star)  [row ON col]",
                self.lowlevel_on_star,
            )
            print_table(
                "Relation: Higher (lowlevel_higher)  [row >= col]", self.lowlevel_higher
            )
            print_table(
                "Relation: Scattered (lowlevel_scattered)  [row scattered-from col]",
                self.lowlevel_scattered,
            )

        for block, color in zip(blocks, colors[: len(blocks)]):
            cx = to_float(self.X(block))
            cy = to_float(self.Y(block))
            cz = to_float(self.Z(block))
            draw_cube(ax, cx, cy, cz, L_val, color=color)
            include_aabb(cx, cy, cz, half_side, half_side, half_side)
            print(str(block), cx, cy, cz)
            ax.text(cx, cy, cz, str(block), fontsize=12)

        # Print relationships among all objects we are visualizing.
        rel_objects = list(blocks)
        rel_labels = [str(b) for b in blocks]

        if extra_blocks:
            for item in extra_blocks:
                if len(item) == 3:
                    block, color, label = item
                elif len(item) == 2:
                    block, label = item
                    color = "black"
                else:
                    block = item[0]
                    color = "black"
                    label = str(block)
                cx = to_float(self.X(block))
                cy = to_float(self.Y(block))
                cz = to_float(self.Z(block))
                draw_cube(ax, cx, cy, cz, L_val, color=color)
                include_aabb(cx, cy, cz, half_side, half_side, half_side)
                print(str(label), cx, cy, cz)
                ax.text(cx, cy, cz, str(label), fontsize=12, color=color)
                rel_objects.append(block)
                rel_labels.append(str(label))

        if rel_objects:
            print_relation_tables(rel_objects, rel_labels)

        if encoded_tube is not None:
            p0_expr, p1_expr, label = encoded_tube
            p0 = tuple(to_float(e) for e in p0_expr)
            p1 = tuple(to_float(e) for e in p1_expr)
            # In encode_collision we use Abs(coord - c(t)) < L, so L is halfwidth.
            draw_encoded_tube(ax, p0, p1, halfwidth=L_val)
            # Tube halfwidth in each dimension is L_val (not L_val/2).
            include_aabb(p0[0], p0[1], p0[2], L_val, L_val, L_val)
            include_aabb(p1[0], p1[1], p1[2], L_val, L_val, L_val)
            mx = 0.5 * (p0[0] + p1[0])
            my = 0.5 * (p0[1] + p1[1])
            mz = 0.5 * (p0[2] + p1[2])
            ax.text(mx, my, mz, str(label), fontsize=12, color="purple")

        # Enforce equal scaling so cubes render as cubes even if X/Y/Z ranges differ.
        if min_x != float("inf"):
            cx = 0.5 * (min_x + max_x)
            cy = 0.5 * (min_y + max_y)
            cz = 0.5 * (min_z + max_z)
            max_range = max(max_x - min_x, max_y - min_y, max_z - min_z)
            if max_range == 0:
                max_range = L_val
            half = 0.5 * max_range
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_zlim(cz - half, cz + half)

        # Newer matplotlib supports this and improves cube appearance further.
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if save_path:
            plt.savefig(save_path)
            print(f"Scene saved to {save_path}")
        else:
            plt.show()

        return ax

    def start_verification(
        self,
        initial_condition: List,
        pickplace_instructions: List,
        constants: List,
        *,
        contract=None,
        noise=None,
        block_v="0",
        timeout_ms=5000,
    ):
        """Check the explicitly declared block contract, frame, and swept geometry."""
        from synthesis.verification_lib.motion_verification import verify_motion_block

        return verify_motion_block(
            initial_condition,
            pickplace_instructions,
            constants,
            contract,
            context=self,
            noise=noise,
            block_v=block_v,
            timeout_ms=timeout_ms,
        )

    def _fresh_skolem_const(self, avoid=()):
        """A Box constant no other term uses, for witnessing an existential."""
        return fresh_const(self.BoxSort, prefix="sk_", avoid=avoid)

    @staticmethod
    def _child_polarities(expr, polarity):
        """Polarity each argument of *expr* is translated under.

        ``True`` positive, ``False`` negative, ``None`` both (as under an
        ``Iff``), where neither quantifier rule is valid.
        """
        num_args = expr.num_args()
        flipped = None if polarity is None else not polarity
        kind = expr.decl().kind()
        if kind == Z3_OP_NOT:
            return [flipped] * num_args
        if kind == Z3_OP_IMPLIES:
            return [flipped] + [polarity] * (num_args - 1)
        if kind == Z3_OP_XOR or (
            kind in (Z3_OP_EQ, Z3_OP_DISTINCT) and is_bool(expr.arg(0))
        ):
            return [None] * num_args
        if kind == Z3_OP_ITE:
            return [None] + [polarity] * (num_args - 1)
        return [polarity] * num_args

    def _translate_expr(
        self, expr, lowlevel_constants, const_map, bindings, polarity=True
    ):
        """Recursively translate a high-level z3 expression to low-level.

        ON_star -> current geometry, ON_star_zero -> frozen entry geometry,
        Higher -> lowlevel_higher,
        Scattered -> lowlevel_scattered. Boolean structure is preserved.

        The result is *asserted* as an assumption, so it has to be implied by the
        original rather than equivalent to it: anything unsatisfiable under a
        weaker assumption is unsatisfiable under the real one. That is what fixes
        the treatment of each quantifier, and it depends on polarity:

        * ``ForAll`` at positive polarity becomes the finite conjunction over
          ``lowlevel_constants``, which the real universally quantified formula
          implies -- the Box sort has no finite domain bound, so
          this is a weakening and not an equivalence.
        * ``Exists`` at positive polarity is Skolemized: fresh constants, one per
          occurrence. Since the enclosing universals have already been expanded
          into separate conjuncts by the time we get here, each instantiation
          gets its own witness, which is what a Skolem *function* of those
          universals would give. This is satisfiability-preserving, so it is
          exact rather than merely sound. The finite disjunction over the named
          constants would instead be *stronger* than the existential -- it
          demands the witness be one of the constants we happened to name -- and
          could make a check come out unsat that is not.

        Negative quantifiers are first dualized: not forall becomes exists not,
        and not exists becomes forall not. The positive rules then apply to the
        negated formula. Mixed polarity (for example Boolean equality) is refused.

        bindings: list where bindings[de_bruijn_index] = concrete lowlevel constant.
        """
        if is_var(expr):
            return bindings[get_var_index(expr)]

        if is_quantifier(expr):
            if polarity is None:
                raise NotImplementedError(
                    f"Quantifier at mixed polarity in a low-level condition: {expr}"
                )
            if polarity is False:
                variables, body = open_quantifier(
                    expr, avoid=(*const_map.values(), *bindings)
                )
                dual = (Exists if expr.is_forall() else ForAll)(variables, Not(body))
                # The surrounding negative position cancels this outer Not.
                # Applying finite expansion directly under Not would strengthen
                # the premise; dualizing first preserves the weakening direction.
                return Not(
                    self._translate_expr(
                        dual, lowlevel_constants, const_map, bindings, True
                    )
                )
            num_vars = expr.num_vars()
            body = expr.body()
            if expr.is_forall():
                conjuncts = []
                for assignment in product(lowlevel_constants, repeat=num_vars):
                    # de Bruijn: var_name(i) has index (num_vars - 1 - i) in body
                    new_prefix = [None] * num_vars
                    for i in range(num_vars):
                        new_prefix[num_vars - 1 - i] = assignment[i]
                    new_bindings = new_prefix + bindings
                    conjuncts.append(
                        self._translate_expr(
                            body, lowlevel_constants, const_map, new_bindings, polarity
                        )
                    )
                return And(*conjuncts)

            witnesses = [
                self._fresh_skolem_const((expr, *const_map.values(), *bindings))
                for _ in range(num_vars)
            ]
            new_prefix = [None] * num_vars
            for i in range(num_vars):
                new_prefix[num_vars - 1 - i] = witnesses[i]
            return self._translate_expr(
                body, lowlevel_constants, const_map, new_prefix + bindings, polarity
            )

        if is_app(expr):
            decl = expr.decl()

            if decl.kind() == Z3_OP_UNINTERPRETED:
                name = decl.name()
                if decl.arity() == 0:
                    if name in const_map:
                        return const_map[name]
                    raise ValueError(f"Unknown constant in translation: {name}")

                children = [
                    self._translate_expr(
                        c, lowlevel_constants, const_map, bindings, polarity
                    )
                    for c in expr.children()
                ]
                if name == "ON_star":
                    return self.lowlevel_on_star(children[0], children[1])
                if name == "ON_star_zero":
                    return self.lowlevel_on_star_zero(children[0], children[1])
                if name == "Higher":
                    return self.lowlevel_higher(children[0], children[1])
                if name == "Scattered":
                    return self.lowlevel_scattered(children[0], children[1])
                raise NotImplementedError(f"Unhandled high-level predicate: {name}")

            if decl.kind() == Z3_OP_EQ and not is_bool(expr.arg(0)):
                children = [
                    self._translate_expr(
                        c, lowlevel_constants, const_map, bindings, polarity
                    )
                    for c in expr.children()
                ]
                if children[0].sort() == self.BoxSort:
                    return self.lowlevel_box_equal(children[0], children[1])
                return decl(*children)

            if decl.kind() == Z3_OP_DISTINCT and not is_bool(expr.arg(0)):
                children = [
                    self._translate_expr(
                        c, lowlevel_constants, const_map, bindings, polarity
                    )
                    for c in expr.children()
                ]
                # Mirror the `==` translation: prefer "distinct position" for box terms.
                # If the children are not box terms, constructing lowlevel_box_equal will
                # raise a sort/type error; in that case, preserve the original Distinct.
                try:
                    if len(children) == 2:
                        return Not(self.lowlevel_box_equal(children[0], children[1]))
                    pairwise = []
                    for i in range(len(children)):
                        for j in range(i + 1, len(children)):
                            pairwise.append(
                                Not(self.lowlevel_box_equal(children[i], children[j]))
                            )
                    return And(*pairwise) if pairwise else BoolVal(True)
                except Exception:
                    return decl(*children)

            child_polarities = self._child_polarities(expr, polarity)
            children = [
                self._translate_expr(c, lowlevel_constants, const_map, bindings, pol)
                for c, pol in zip(expr.children(), child_polarities)
            ]
            return decl(*children)

        return expr

    def translate_exact(self, expr, const_map, bindings=()):
        """Interpret a postcondition/WP exactly, preserving quantified objects.

        Unlike assumption weakening, this is safe inside equivalence/negation.
        Quantified NRA can be inconclusive; callers must preserve that verdict.
        """
        if is_var(expr):
            return bindings[get_var_index(expr)]
        if is_quantifier(expr):
            variables = [
                fresh_const(
                    self.BoxSort,
                    prefix="effect",
                    avoid=(expr, *const_map.values(), *bindings),
                )
                for _ in range(expr.num_vars())
            ]
            body = self.translate_exact(
                expr.body(), const_map, tuple(reversed(variables)) + tuple(bindings)
            )
            return (ForAll if expr.is_forall() else Exists)(variables, body)
        if not is_app(expr):
            return expr
        decl = expr.decl()
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.arity() == 0:
            return const_map[str(decl.name())]
        children = [
            self.translate_exact(c, const_map, bindings) for c in expr.children()
        ]
        if decl.kind() == Z3_OP_UNINTERPRETED:
            relations = {
                "ON_star": self.lowlevel_on_star,
                "ON_star_zero": self.lowlevel_on_star_zero,
                "Higher": self.lowlevel_higher,
                "Scattered": self.lowlevel_scattered,
            }
            if str(decl.name()) not in relations:
                raise NotImplementedError(f"Unsupported exact relation: {decl.name()}")
            return relations[str(decl.name())](*children)
        if decl.kind() == Z3_OP_EQ and children and children[0].sort() == self.BoxSort:
            return self.lowlevel_box_equal(*children)
        if (
            decl.kind() == Z3_OP_DISTINCT
            and children
            and children[0].sort() == self.BoxSort
        ):
            return And(
                *(
                    Not(self.lowlevel_box_equal(a, b))
                    for i, a in enumerate(children)
                    for b in children[i + 1 :]
                )
            )
        return decl(*children)

    def translate_condition(
        self, s: Solver, constants: List, conditions: List, *, track=True
    ):
        const_map = {str(c): self.get_consts(str(c)) for c in constants}
        lowlevel_constants = list(const_map.values())
        sym = self.get_consts("sym")
        lowlevel_constants.append(sym)

        for idx, condition in enumerate(conditions):
            print(f"translating condition {idx}, {condition}")
            translated = self._translate_expr(
                condition, lowlevel_constants, const_map, []
            )
            if track:
                s.assert_and_track(translated, f"condition_{idx}")
            else:
                s.add(translated)

        return const_map
