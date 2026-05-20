import argparse
from pathlib import Path
from typing import Optional, Union

from z3 import And, Const, Consts, ExprRef, ForAll, If, Implies, Not

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api.instructions import GoalAssign, MarkGoal, MoveDown, MoveRight, While
from synthesis.api.program import Program
from synthesis.inference_lib.inference import (
    ProvenancedClause,
    run_2d_inner_loop_example,
    run_2d_outer_loop_example,
)


def _mk_mark_fn(context, mark_fn):
    return mark_fn if mark_fn is not None else (lambda t: context.Mark(t))


def _outer_template(context, head, i0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_I1", context.GoalSort)
    scanned = If(
        i0 == context.null,
        context._dr_reach(head, x),
        And(context._flat_between(head, x, i0), x != i0),
    )
    return And(
        head != context.null,
        context.l0(head) == head,
        Implies(i0 != context.null, context.d_star(head, i0)),
        ForAll([x], Implies(scanned, m(x))),
    )


def _inner_template(context, head, i0, j0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_I2", context.GoalSort)
    inner_scanned = If(
        j0 == context.null,
        context.r_star(i0, x),
        And(context.r_star(i0, x), context._f_plus(context.r_star, x, j0)),
    )
    return And(
        head != context.null,
        context.l0(head) == head,
        context.d_star(head, i0),
        Implies(j0 != context.null, context.r_star(i0, j0)),
        ForAll([x], Implies(And(context._flat_between(head, x, i0), x != i0), m(x))),
        ForAll([x], Implies(inner_scanned, m(x))),
    )


def precondition(context, h0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_P", context.GoalSort)
    return And(
        h0 != context.null,
        context.l0(h0) == h0,
        ForAll([x], Implies(x != context.null, context._dr_reach(h0, x))),
        ForAll([x], Not(m(x))),
    )


def postcondition(context, h0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_Q", context.GoalSort)
    return ForAll([x], Implies(context._dr_reach(h0, x), m(x)))


def _build_program(context, h, i, j, outer_inv, inner_inv):
    instructions = [
        GoalAssign("i", "h"),
        While(
            instantiated_cond=i != context.null,
            guard_exists_vars=[],
            body=[
                GoalAssign("j", "i"),
                While(
                    instantiated_cond=j != context.null,
                    guard_exists_vars=[],
                    body=[MarkGoal("j"), MoveRight("j")],
                    invariant=inner_inv,
                ),
                MoveDown("i"),
            ],
            invariant=outer_inv,
        ),
    ]
    return Program(2, instructions=instructions)


def infer_outer_loop_invariant(context) -> list[ProvenancedClause]:
    """
    Infer only the outer-loop invariant (proposal+validation in inference_lib).

    Inner-loop invariant is intentionally not inferred here so callers can debug
    the outer invariant against a known-good handwritten inner (_inner_template).
    """
    _outer_and, outer_clauses = run_2d_outer_loop_example(context)
    # We return the clause list so verification can report provenance by index.
    return outer_clauses


def infer_inner_loop_invariant(context) -> list[ProvenancedClause]:
    """
    Infer only the inner-loop invariant (proposal+validation in inference_lib).

    Outer-loop invariant is intentionally not inferred here so callers can debug
    the inner invariant against a known-good handwritten outer (_outer_template).
    """
    _inner_and, inner_clauses = run_2d_inner_loop_example(context)
    return inner_clauses


def verify_2d_program_with_learned_invariant(
    use_hardcoded_invariant: bool = False,
    counterexample_image_dir: Optional[Union[str, Path]] = None,
    num_goals: Optional[int] = None,
):
    """
    Verify nested while-loop mark-all using program-level VC generation.

    Default path: learned outer invariant + handwritten inner (_inner_template).
    With --use-hardcoded-invariant: both loops use the original templates.
    If ``num_goals`` is set, Goal sort is a finite ``EnumSort`` of ``null`` and
    ``g0``..``g{num_goals-1}``; otherwise Goal is an uninterpreted sort.
    """
    if num_goals is not None and num_goals < 0:
        raise ValueError("num_goals must be non-negative")
    if num_goals is not None:
        context = highlevel_verification_lib.HighLevelContext(
            mode="enum",
            num_goals=num_goals,
            verification_mode="goals",
        )
    else:
        context = highlevel_verification_lib.HighLevelContext(
            mode="declare",
            verification_mode="goals",
        )

    h, i, j = Consts("h i j", context.GoalSort)
    if use_hardcoded_invariant:
        outer_inv = _outer_template(context, h, i)
        inner_inv = _inner_template(context, h, i, j)
    else:
        outer_inv = infer_outer_loop_invariant(context)
        inner_inv = infer_inner_loop_invariant(context)
        # inner_inv = _inner_template(context, h, i, j)
        # Debug check: compare learned outer vs handwritten outer as plain Z3 formulas.
        # print("checking learned outer vs handwritten outer")
        outer_inv_z3 = And(*[c.expr for c in outer_inv])
        context.check_satisfiable_with_pq(
            p=outer_inv_z3, q=_outer_template(context, h, i)
        )

        print("checking learned inner vs handwritten inner")
        inner_inv_z3 = And(*[c.expr for c in inner_inv])
        context.check_satisfiable_with_pq(
            p=inner_inv_z3, q=_inner_template(context, h, i, j)
        )

    program = _build_program(context, h, i, j, outer_inv, inner_inv)
    pre = precondition(context, h)
    post = postcondition(context, h)
    return bool(
        program.highlevel_verification(
            pre,
            post,
            context=context,
            counterexample_image_dir=counterexample_image_dir,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-hardcoded-invariant",
        action="store_true",
        help="Use handwritten outer and inner templates; otherwise outer is learned, inner is handwritten.",
    )
    parser.add_argument(
        "--counterexample-dir",
        default="highlevel_counterexamples",
        metavar="DIR",
        help=(
            "Where to save Goal-mode counterexample PNGs when a VC check fails. "
            "Pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--num-goals",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, use a finite Goal enum: null and g0..gN-1 (enum mode). "
            "Omit for an uninterpreted Goal sort (declare mode)."
        ),
    )
    args = parser.parse_args()
    ce = args.counterexample_dir.strip() or None
    success = verify_2d_program_with_learned_invariant(
        use_hardcoded_invariant=args.use_hardcoded_invariant,
        counterexample_image_dir=ce,
        num_goals=args.num_goals,
    )
    print(f"overall_ok: {success}")
