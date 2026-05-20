"""
Bounded model checking for the pick / move / release fragment of
``synthesis.api.instructions``.

Supported concrete instruction classes:

* ``Pick``, ``PickByName``
* ``Move``, ``MoveByName``
* ``Release``, ``ReleaseByName`` (physical “place / let go”; there are no
  separate ``Place`` / ``PlaceByName`` classes in the DSL).

Programs must be straight-line (no ``While``). A program must use **either**
only ByName instructions (``PickByName``, ``MoveByName``, ``ReleaseByName``) or
only id-based ones (``Pick``, ``Move``, ``Release``); mixing raises
``ValueError``.

The set of blocks is **inferred** from operands: each distinct name or each
distinct integer id is a different block. Id-based programs use trace keys
``str(raw_id)`` (e.g. ``"0"``, ``"2"``) in sorted id order. Name-based programs
use sorted unique strings as keys.

**Solve** mode (:func:`bmc_feasible`, :func:`bmc_solve`): ``Move*`` offsets and
``Release*`` ``target_z_offset`` are fresh **existential** Z3 reals; omitted
instruction parameters (``Parameter.val is None``) are **not** read.

**Verify** mode (:func:`bmc_verify`, :func:`bmc_verify_solve`): offsets are
fixed to ``RealVal`` from each instruction's concrete ``Parameter.val``; any
offset still ``None`` raises ``ValueError``. The check is **obligation**
semantics: let ``P`` be initial ∧ extra ∧ transition constraints (and optional
``¬goal@0``). Verification returns true iff ``P`` is satisfiable (the program
can run from the initial facts) and ``P ∧ ¬goal@T`` is **UNSAT** — i.e.
``P ⇒ goal@T`` (no trajectory allowed by ``P`` violates the goal at the final
step). This matches ``Implies(P, goal)`` / no counterexample, not
``SAT(P ∧ goal)``.

Typical stacking goals ``ON(upper, lower)`` use the same geometry as
``synthesis.util.on.on`` via :func:`z3_on`, or the helpers
:func:`goal_on_box_ids` (id-only programs, integer gym ids) and
:func:`goal_on_trace_keys` (ByName programs, exact operand strings such as
``\"b\"`` and ``\"b_prime\"``).

By default, :func:`bmc_feasible` / :func:`bmc_solve` also assert that the
``goal`` formula is **false at time** ``0`` (rewritten from its usual
evaluation at ``sym.T`` via :func:`instantiate_goal_at_time`), so the goal is
interpreted as something to **achieve** along the trace, not already satisfied
initially. Set ``assume_goal_false_at_start=False`` to disable. When
``sym.T == 0`` (empty program), this guard is skipped.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import z3

from synthesis.api.instructions import (
    Instruction,
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    Seq,
)
from synthesis.util.on import z3_on

ProgramPart = Union[Instruction, Seq]
ProgramInput = Union[Sequence[ProgramPart], ProgramPart]


def flatten_program(program: ProgramInput) -> List[Instruction]:
    """Flatten a program that may mix lists and binary ``Seq`` trees."""

    def walk(p: ProgramInput) -> List[Instruction]:
        if isinstance(p, Seq):
            return walk(p.s1) + walk(p.s2)
        if (
            isinstance(p, Sequence)
            and not isinstance(p, (str, bytes))
            and not isinstance(p, Instruction)
        ):
            out: List[Instruction] = []
            for x in p:
                out.extend(walk(x))  # type: ignore[arg-type]
            return out
        if isinstance(p, Instruction):
            return [p]
        raise TypeError(f"Unsupported program fragment: {type(p)}")

    return walk(program)


_BMC_INSTRUCTION_TYPES = (Pick, PickByName, Move, MoveByName, Release, ReleaseByName)


def _check_program_supported(instrs: Sequence[Instruction]) -> None:
    for i, ins in enumerate(instrs):
        if not isinstance(ins, _BMC_INSTRUCTION_TYPES):
            raise TypeError(
                f"BMC step {i}: unsupported instruction {type(ins).__name__}. "
                f"Allowed: {tuple(c.__name__ for c in _BMC_INSTRUCTION_TYPES)}."
            )


_BY_NAME_TYPES = (PickByName, MoveByName, ReleaseByName)
_BY_ID_TYPES = (Pick, Move, Release)


def infer_block_layout(program: ProgramInput) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    """
    Derive ``(block_names, name_to_box_id)`` from the program alone.

    ``block_names`` is ordered; ``name_to_box_id`` maps each trace key (a block
    name string, or ``str(int_id)`` in id-only mode) to its compact index
    ``0 .. len(block_names)-1``.

    Raises ``ValueError`` if the program mixes ByName and id-based instructions.
    """
    flat = flatten_program(program)
    _check_program_supported(flat)
    return _infer_block_universe(flat)


def _infer_block_universe(
    flat: Sequence[Instruction],
) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    has_name = any(isinstance(ins, _BY_NAME_TYPES) for ins in flat)
    has_id = any(isinstance(ins, _BY_ID_TYPES) for ins in flat)
    if has_name and has_id:
        raise ValueError(
            "BMC program mixes ByName instructions with id-based ones. "
            "Use only PickByName/MoveByName/ReleaseByName, or only Pick/Move/Release."
        )
    if not flat or (not has_name and not has_id):
        return (), {}

    if has_name:
        names: Set[str] = set()
        for ins in flat:
            if isinstance(ins, PickByName):
                names.add(ins.grab_box_name)
            elif isinstance(ins, MoveByName):
                names.update(
                    (
                        ins.target_box_name_x,
                        ins.target_box_name_y,
                        ins.target_box_name_z,
                    )
                )
            elif isinstance(ins, ReleaseByName):
                names.add(ins.release_box_name)
        ordered = tuple(sorted(names))
        return ordered, {n: i for i, n in enumerate(ordered)}

    ids: Set[int] = set()
    for ins in flat:
        if isinstance(ins, Pick):
            ids.add(int(ins.grab_box_id))
        elif isinstance(ins, Move):
            ids.update(
                (
                    int(ins.target_box_id_x),
                    int(ins.target_box_id_y),
                    int(ins.target_box_id_z),
                )
            )
        elif isinstance(ins, Release):
            ids.add(int(ins.release_box_id))
    ordered_ids = tuple(sorted(ids))
    block_names = tuple(str(rid) for rid in ordered_ids)
    name_to_box_id = {str(rid): j for j, rid in enumerate(ordered_ids)}
    return block_names, name_to_box_id


@dataclass(frozen=True)
class BMCTraceSymbols:
    """Per-time-step symbols for a BMC encoding (``t = 0 .. T`` inclusive)."""

    T: int
    block_names: Tuple[str, ...]
    ee_x: List[z3.ArithRef]
    ee_y: List[z3.ArithRef]
    ee_z: List[z3.ArithRef]
    holding: List[z3.ExprRef]
    bx: Dict[str, List[z3.ArithRef]]
    by: Dict[str, List[z3.ArithRef]]
    bz: Dict[str, List[z3.ArithRef]]
    ObjSort: z3.SortRef
    NONE: z3.ExprRef
    block_consts: Dict[str, z3.ExprRef]


_bmc_enum_sort_id = itertools.count()


def _make_holding_sort(
    block_names: Sequence[str],
) -> Tuple[z3.SortRef, z3.ExprRef, Dict[str, z3.ExprRef]]:
    names = ["none"] + [str(b) for b in block_names]
    sort_name = f"BMC_Obj_{next(_bmc_enum_sort_id)}"
    Obj, consts = z3.EnumSort(sort_name, names)
    none = consts[0]
    block_consts = {str(b): consts[i + 1] for i, b in enumerate(block_names)}
    return Obj, none, block_consts


def build_trace_symbols(block_names: Sequence[str], num_steps: int) -> BMCTraceSymbols:
    names = tuple(str(b) for b in block_names)
    T = num_steps
    Obj, NONE, block_consts = _make_holding_sort(names)
    ee_x = [z3.Real(f"bmc_ee_x_{t}") for t in range(T + 1)]
    ee_y = [z3.Real(f"bmc_ee_y_{t}") for t in range(T + 1)]
    ee_z = [z3.Real(f"bmc_ee_z_{t}") for t in range(T + 1)]
    holding = [z3.Const(f"bmc_holding_{t}", Obj) for t in range(T + 1)]
    bx: Dict[str, List[z3.ArithRef]] = {}
    by: Dict[str, List[z3.ArithRef]] = {}
    bz: Dict[str, List[z3.ArithRef]] = {}
    for b in names:
        bx[b] = [z3.Real(f"bmc_{b}_x_{t}") for t in range(T + 1)]
        by[b] = [z3.Real(f"bmc_{b}_y_{t}") for t in range(T + 1)]
        bz[b] = [z3.Real(f"bmc_{b}_z_{t}") for t in range(T + 1)]
    return BMCTraceSymbols(
        T=T,
        block_names=names,
        ee_x=ee_x,
        ee_y=ee_y,
        ee_z=ee_z,
        holding=holding,
        bx=bx,
        by=by,
        bz=bz,
        ObjSort=Obj,
        NONE=NONE,
        block_consts=block_consts,
    )


# Initial / extra facts: fixed list, or a factory using the **solver's** ``sym``
# (callers must not build constraints on a different ``BMCTraceSymbols``).
ConstraintSpec = Optional[
    Union[Iterable[z3.BoolRef], Callable[[BMCTraceSymbols], Iterable[z3.BoolRef]]]
]


def _expand_constraints(sym: BMCTraceSymbols, spec: ConstraintSpec) -> List[z3.BoolRef]:
    if spec is None:
        return []
    if callable(spec):
        return list(spec(sym))
    return list(spec)


def goal_on_trace_keys(
    sym: BMCTraceSymbols,
    upper: str,
    lower: str,
    *,
    t: Optional[int] = None,
) -> z3.BoolRef:
    """
    Stacking goal ``ON(upper, lower)`` using **trace keys** (entries of
    ``sym.block_names``): for ByName programs these are the exact strings used
    in ``PickByName`` / ``MoveByName`` / ``ReleaseByName`` (e.g. ``\"b\"`` and
    ``\"b_prime\"``, not Z3 constant names unless they match those strings).
    For id-only programs, keys are ``str(int_id)`` (e.g. ``\"2\"``, ``\"1\"``).

    Raises ``KeyError`` if a key is not present in the trace.
    """
    for k, role in ((upper, "upper"), (lower, "lower")):
        if k not in sym.bx:
            raise KeyError(
                f"goal_on_trace_keys: unknown {role} block key {k!r}; "
                f"sym.block_names is {sym.block_names}"
            )
    ti = sym.T if t is None else t
    return z3_on(
        sym.bx[upper][ti],
        sym.by[upper][ti],
        sym.bz[upper][ti],
        sym.bx[lower][ti],
        sym.by[lower][ti],
        sym.bz[lower][ti],
    )


def goal_on_box_ids(
    sym: BMCTraceSymbols,
    upper_id: int,
    lower_id: int,
    *,
    t: Optional[int] = None,
) -> z3.BoolRef:
    """
    Same as :func:`goal_on_trace_keys` but takes **integer box ids** as in
    ``Pick`` / ``Move`` / ``Release`` (mapped to ``str(id)`` trace keys).
    Intended for id-only programs.
    """
    return goal_on_trace_keys(sym, str(int(upper_id)), str(int(lower_id)), t=t)


def instantiate_goal_at_time(
    sym: BMCTraceSymbols, goal_expr: z3.BoolRef, t_from: int, t_to: int
) -> z3.BoolRef:
    """
    Rewire every EE / holding / block coordinate at ``t_from`` in ``goal_expr``
    to the corresponding variable at ``t_to`` (same keys, different time index).

    Used so a goal normally stated at ``sym.T`` can be evaluated at the initial
    step ``0``.
    """
    if t_from == t_to:
        return goal_expr
    pairs: List[Tuple[z3.ExprRef, z3.ExprRef]] = [
        (sym.ee_x[t_from], sym.ee_x[t_to]),
        (sym.ee_y[t_from], sym.ee_y[t_to]),
        (sym.ee_z[t_from], sym.ee_z[t_to]),
        (sym.holding[t_from], sym.holding[t_to]),
    ]
    for b in sym.block_names:
        pairs.append((sym.bx[b][t_from], sym.bx[b][t_to]))
        pairs.append((sym.by[b][t_from], sym.by[b][t_to]))
        pairs.append((sym.bz[b][t_from], sym.bz[b][t_to]))
    return z3.substitute(goal_expr, pairs)


def _frame_block(sym: BMCTraceSymbols, block: str, t: int) -> z3.BoolRef:
    return z3.And(
        sym.bx[block][t + 1] == sym.bx[block][t],
        sym.by[block][t + 1] == sym.by[block][t],
        sym.bz[block][t + 1] == sym.bz[block][t],
    )


def _frame_all(sym: BMCTraceSymbols, t: int) -> List[z3.BoolRef]:
    return [_frame_block(sym, b, t) for b in sym.block_names]


def _encode_pick(sym: BMCTraceSymbols, t: int, box_id: int) -> z3.BoolRef:
    if not 0 <= box_id < len(sym.block_names):
        raise IndexError(
            f"Pick box id {box_id} out of range for {len(sym.block_names)} blocks."
        )
    name = sym.block_names[box_id]
    cons: List[z3.BoolRef] = [
        sym.holding[t] == sym.NONE,
        sym.ee_x[t] == sym.bx[name][t],
        sym.ee_y[t] == sym.by[name][t],
        sym.holding[t + 1] == sym.block_consts[name],
        sym.ee_x[t + 1] == sym.ee_x[t],
        sym.ee_y[t + 1] == sym.ee_y[t],
        sym.ee_z[t + 1] == sym.ee_z[t],
    ]
    cons.extend(_frame_all(sym, t))
    return z3.And(*cons)


def _move_target(
    sym: BMCTraceSymbols,
    t: int,
    ix: int,
    iy: int,
    iz: int,
    ox: z3.ArithRef,
    oy: z3.ArithRef,
    oz: z3.ArithRef,
) -> Tuple[z3.ArithRef, z3.ArithRef, z3.ArithRef]:
    nx = sym.block_names[ix]
    ny = sym.block_names[iy]
    nz = sym.block_names[iz]
    return (
        sym.bx[nx][t] + ox,
        sym.by[ny][t] + oy,
        sym.bz[nz][t] + oz,
    )


def _encode_move(
    sym: BMCTraceSymbols,
    t: int,
    ix: int,
    iy: int,
    iz: int,
    ox: z3.ArithRef,
    oy: z3.ArithRef,
    oz: z3.ArithRef,
) -> z3.BoolRef:
    tx, ty, tz = _move_target(sym, t, ix, iy, iz, ox, oy, oz)
    moved: List[z3.BoolRef] = [
        sym.ee_x[t + 1] == tx,
        sym.ee_y[t + 1] == ty,
        sym.ee_z[t + 1] == tz,
    ]
    for j, bj in enumerate(sym.block_names):
        held = sym.holding[t] == sym.block_consts[bj]
        moved.append(
            z3.If(
                held,
                z3.And(
                    sym.bx[bj][t + 1] == tx,
                    sym.by[bj][t + 1] == ty,
                    sym.bz[bj][t + 1] == tz,
                ),
                _frame_block(sym, bj, t),
            )
        )
    moved.append(sym.holding[t + 1] == sym.holding[t])
    return z3.And(*moved)


def _encode_release(sym: BMCTraceSymbols, t: int, z_off: z3.ArithRef) -> z3.BoolRef:
    # ``z_off`` is kept for API parity with ``Release.target_z_offset``; the
    # simple BMC model fixes geometry at release (see user example).
    _ = z_off
    return z3.And(
        sym.holding[t + 1] == sym.NONE,
        sym.ee_x[t + 1] == sym.ee_x[t],
        sym.ee_y[t + 1] == sym.ee_y[t],
        sym.ee_z[t + 1] == sym.ee_z[t],
        z3.And(*_frame_all(sym, t)),
    )


def encode_step(
    sym: BMCTraceSymbols,
    t: int,
    instr: Instruction,
    *,
    name_to_box_id: Mapping[str, int],
    offset_vars: Mapping[Tuple[int, str], z3.ArithRef],
) -> z3.BoolRef:
    """Encode ``instr`` as the transition from time ``t`` to ``t+1``."""
    key_ox = (t, "ox")
    key_oy = (t, "oy")
    key_oz = (t, "oz")
    key_rz = (t, "rz")

    if isinstance(instr, Pick):
        k = str(int(instr.grab_box_id))
        if k not in name_to_box_id:
            raise KeyError(
                f"Pick: box id {instr.grab_box_id!r} not in inferred layout keys."
            )
        return _encode_pick(sym, t, int(name_to_box_id[k]))
    if isinstance(instr, PickByName):
        if instr.grab_box_name not in name_to_box_id:
            raise KeyError(
                f"PickByName: name {instr.grab_box_name!r} not in inferred layout "
                f"(known: {sorted(name_to_box_id)})."
            )
        return _encode_pick(sym, t, int(name_to_box_id[instr.grab_box_name]))
    if isinstance(instr, Move):
        ox = offset_vars[key_ox]
        oy = offset_vars[key_oy]
        oz = offset_vars[key_oz]
        mx = str(int(instr.target_box_id_x))
        my = str(int(instr.target_box_id_y))
        mz = str(int(instr.target_box_id_z))
        for key, label in ((mx, "x"), (my, "y"), (mz, "z")):
            if key not in name_to_box_id:
                raise KeyError(
                    f"Move: target_box_id_{label}={key!r} not in inferred layout keys."
                )
        ix = int(name_to_box_id[mx])
        iy = int(name_to_box_id[my])
        iz = int(name_to_box_id[mz])
        return _encode_move(sym, t, ix, iy, iz, ox, oy, oz)
    if isinstance(instr, MoveByName):
        m = name_to_box_id
        for nm, label in (
            (instr.target_box_name_x, "x"),
            (instr.target_box_name_y, "y"),
            (instr.target_box_name_z, "z"),
        ):
            if nm not in m:
                raise KeyError(
                    f"MoveByName: target_box_name_{label}={nm!r} not in inferred layout "
                    f"(known: {sorted(m)})."
                )
        ix = int(m[instr.target_box_name_x])
        iy = int(m[instr.target_box_name_y])
        iz = int(m[instr.target_box_name_z])
        ox = offset_vars[key_ox]
        oy = offset_vars[key_oy]
        oz = offset_vars[key_oz]
        return _encode_move(sym, t, ix, iy, iz, ox, oy, oz)
    if isinstance(instr, Release):
        return _encode_release(sym, t, offset_vars[key_rz])
    if isinstance(instr, ReleaseByName):
        return _encode_release(sym, t, offset_vars[key_rz])
    raise TypeError(f"encode_step: unsupported instruction {type(instr).__name__}")


def _bmc_setup(program: ProgramInput, offset_mode: str) -> Tuple[
    BMCTraceSymbols,
    List[Instruction],
    Dict[Tuple[int, str], z3.ArithRef],
    Mapping[str, int],
]:
    flat = flatten_program(program)
    _check_program_supported(flat)
    block_names, name_to_box_id = _infer_block_universe(flat)
    sym = build_trace_symbols(block_names, len(flat))
    if offset_mode == "verify":
        off = _collect_offset_constants(flat)
    elif offset_mode == "solve":
        off = _collect_offset_variables(flat)
    else:
        raise ValueError(
            f"offset_mode must be 'solve' or 'verify', got {offset_mode!r}"
        )
    return sym, flat, off, name_to_box_id


def _bmc_add_body(
    s: z3.Solver,
    sym: BMCTraceSymbols,
    flat: Sequence[Instruction],
    name_to_box_id: Mapping[str, int],
    off: Mapping[Tuple[int, str], z3.ArithRef],
    *,
    initial_constraints: ConstraintSpec,
    extra_constraints: ConstraintSpec,
) -> None:
    s.add(*_expand_constraints(sym, initial_constraints))
    s.add(*_expand_constraints(sym, extra_constraints))
    for t, instr in enumerate(flat):
        s.add(
            encode_step(sym, t, instr, name_to_box_id=name_to_box_id, offset_vars=off)
        )


def _bmc_build_and_check_solve(
    program: ProgramInput,
    goal: Callable[[BMCTraceSymbols], z3.BoolRef],
    *,
    initial_constraints: ConstraintSpec,
    extra_constraints: ConstraintSpec,
    assume_goal_false_at_start: bool,
    solver: Optional[z3.Solver],
) -> Tuple[BMCTraceSymbols, z3.Solver, z3.CheckSatResult]:
    sym, flat, off, name_to_box_id = _bmc_setup(program, "solve")
    s = solver if solver is not None else z3.Solver()
    _bmc_add_body(
        s,
        sym,
        flat,
        name_to_box_id,
        off,
        initial_constraints=initial_constraints,
        extra_constraints=extra_constraints,
    )
    g_final = goal(sym)
    if assume_goal_false_at_start and sym.T > 0:
        g0 = instantiate_goal_at_time(sym, g_final, sym.T, 0)
        s.add(z3.Not(g0))
    s.add(g_final)
    r = s.check()
    return sym, s, r


def _bmc_build_and_check_verify(
    program: ProgramInput,
    goal: Callable[[BMCTraceSymbols], z3.BoolRef],
    *,
    initial_constraints: ConstraintSpec,
    extra_constraints: ConstraintSpec,
    assume_goal_false_at_start: bool,
    solver: Optional[z3.Solver],
) -> Tuple[BMCTraceSymbols, z3.Solver, z3.CheckSatResult, z3.CheckSatResult]:
    """Return ``(sym, s_neg_goal, r_body, r_neg)``: ``r_body`` is ``P``; ``r_neg`` is ``P ∧ ¬goal``."""
    sym, flat, off, name_to_box_id = _bmc_setup(program, "verify")
    g_final = goal(sym)

    s_body = solver if solver is not None else z3.Solver()
    _bmc_add_body(
        s_body,
        sym,
        flat,
        name_to_box_id,
        off,
        initial_constraints=initial_constraints,
        extra_constraints=extra_constraints,
    )
    if assume_goal_false_at_start and sym.T > 0:
        g0 = instantiate_goal_at_time(sym, g_final, sym.T, 0)
        s_body.add(z3.Not(g0))
    r_body = s_body.check()

    s_cex = z3.Solver()
    for a in s_body.assertions():
        s_cex.add(a)
    s_cex.add(z3.Not(g_final))
    r_cex = s_cex.check()
    return sym, s_cex, r_body, r_cex


def _collect_offset_variables(
    flat_program: Sequence[Instruction],
) -> Dict[Tuple[int, str], z3.ArithRef]:
    """One Z3 real per trainable offset slot, keyed by (step_index, field)."""
    out: Dict[Tuple[int, str], z3.ArithRef] = {}
    for t, instr in enumerate(flat_program):
        if isinstance(instr, (Move, MoveByName)):
            out[(t, "ox")] = z3.Real(f"bmc_move_{t}_ox")
            out[(t, "oy")] = z3.Real(f"bmc_move_{t}_oy")
            out[(t, "oz")] = z3.Real(f"bmc_move_{t}_oz")
        if isinstance(instr, (Release, ReleaseByName)):
            out[(t, "rz")] = z3.Real(f"bmc_release_{t}_zoff")
    return out


def _collect_offset_constants(
    flat_program: Sequence[Instruction],
) -> Dict[Tuple[int, str], z3.ArithRef]:
    """Fixed ``RealVal`` offsets from each instruction's concrete ``Parameter.val``."""
    out: Dict[Tuple[int, str], z3.ArithRef] = {}
    for t, instr in enumerate(flat_program):
        if isinstance(instr, (Move, MoveByName)):
            ox = instr.target_offset[0].concrete_float(
                f"Move step {t} target_offset[0] (x)"
            )
            oy = instr.target_offset[1].concrete_float(
                f"Move step {t} target_offset[1] (y)"
            )
            oz = instr.target_offset[2].concrete_float(
                f"Move step {t} target_offset[2] (z)"
            )
            out[(t, "ox")] = z3.RealVal(ox)
            out[(t, "oy")] = z3.RealVal(oy)
            out[(t, "oz")] = z3.RealVal(oz)
        if isinstance(instr, (Release, ReleaseByName)):
            rz = instr.target_z_offset.concrete_float(f"Release step {t} target_z")
            out[(t, "rz")] = z3.RealVal(rz)
    return out


def bmc_feasible(
    program: ProgramInput,
    goal: Callable[[BMCTraceSymbols], z3.BoolRef],
    *,
    initial_constraints: ConstraintSpec = None,
    extra_constraints: ConstraintSpec = None,
    assume_goal_false_at_start: bool = True,
    solver: Optional[z3.Solver] = None,
) -> bool:
    """
    Return ``True`` iff there exist **trajectory values** (and existential move /
    release offset reals **independent** of the instruction objects' stored
    parameters, including omitted ``Parameter.val is None`` slots)
    satisfying ``initial_constraints``, the instruction semantics, and
    ``goal(sym)`` at the final time index ``sym.T``.

    When ``assume_goal_false_at_start`` is true and the program has at least
    one step (``sym.T > 0``), the solver also requires ``goal`` evaluated at
    time ``0`` to be false (see module docstring).

    ``initial_constraints`` / ``extra_constraints`` may be a list of formulas
    **or** ``lambda sym: [...]`` so facts use the same ``sym`` instance the
    solver builds (recommended for :func:`default_table_initial_state`).

    Blocks and trace keys are inferred from the program; see
    :func:`infer_block_layout`.
    """
    _sym, _s, r = _bmc_build_and_check_solve(
        program,
        goal,
        initial_constraints=initial_constraints,
        extra_constraints=extra_constraints,
        assume_goal_false_at_start=assume_goal_false_at_start,
        solver=solver,
    )
    return r == z3.sat


def bmc_solve(
    program: ProgramInput,
    goal: Callable[[BMCTraceSymbols], z3.BoolRef],
    *,
    initial_constraints: ConstraintSpec = None,
    extra_constraints: ConstraintSpec = None,
    assume_goal_false_at_start: bool = True,
    solver: Optional[z3.Solver] = None,
) -> Tuple[bool, Optional[z3.ModelRef], BMCTraceSymbols]:
    """Like ``bmc_feasible`` but returns ``(sat?, model, sym)``."""
    sym, s, r = _bmc_build_and_check_solve(
        program,
        goal,
        initial_constraints=initial_constraints,
        extra_constraints=extra_constraints,
        assume_goal_false_at_start=assume_goal_false_at_start,
        solver=solver,
    )
    if r == z3.sat:
        return True, s.model(), sym
    return False, None, sym


def bmc_verify(
    program: ProgramInput,
    goal: Callable[[BMCTraceSymbols], z3.BoolRef],
    *,
    initial_constraints: ConstraintSpec = None,
    extra_constraints: ConstraintSpec = None,
    assume_goal_false_at_start: bool = True,
    solver: Optional[z3.Solver] = None,
) -> bool:
    """
    **Verify** mode (obligation): offsets are fixed to concrete ``Parameter.val``
    (raises if any required offset is still ``None``).

    Let ``P`` be initial ∧ extra ∧ transitions (and optional ``¬goal@0``). Return
    ``True`` iff ``P`` is satisfiable and ``P ∧ ¬goal@T`` is **UNSAT** — every
    run allowed by ``P`` satisfies the goal at the final time (equivalently
    ``Implies(P, goal@T)`` in the SMT sense). If ``P`` is UNSAT, returns
    ``False`` (program not executable from the given initials).
    """
    sym, _, r_body, r_neg = _bmc_build_and_check_verify(
        program,
        goal,
        initial_constraints=initial_constraints,
        extra_constraints=extra_constraints,
        assume_goal_false_at_start=assume_goal_false_at_start,
        solver=solver,
    )
    if r_body != z3.sat:
        return False
    return r_neg == z3.unsat


def bmc_verify_solve(
    program: ProgramInput,
    goal: Callable[[BMCTraceSymbols], z3.BoolRef],
    *,
    initial_constraints: ConstraintSpec = None,
    extra_constraints: ConstraintSpec = None,
    assume_goal_false_at_start: bool = True,
    solver: Optional[z3.Solver] = None,
) -> Tuple[bool, Optional[z3.ModelRef], BMCTraceSymbols]:
    """
    Like :func:`bmc_verify` but returns ``(proved?, model, sym)``.

    * ``proved`` is ``True`` iff verification succeeds (``P ∧ ¬goal`` UNSAT).
    * If ``P`` is UNSAT, returns ``(False, None, sym)``.
    * If ``P ∧ ¬goal`` is SAT, returns ``(False, counterexample_model, sym)``.
    """
    sym, s_cex, r_body, r_neg = _bmc_build_and_check_verify(
        program,
        goal,
        initial_constraints=initial_constraints,
        extra_constraints=extra_constraints,
        assume_goal_false_at_start=assume_goal_false_at_start,
        solver=solver,
    )
    if r_body != z3.sat:
        return False, None, sym
    if r_neg == z3.unsat:
        return True, None, sym
    if r_neg == z3.sat:
        return False, s_cex.model(), sym
    return False, None, sym


def default_table_initial_state(
    sym: BMCTraceSymbols,
    *,
    xy_positions: Mapping[str, Tuple[float, float]],
    table_z: float = 0.0,
    ee_home: Tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> List[z3.BoolRef]:
    """Convenience: blocks on a table grid, gripper empty at ``ee_home``.

    Keys of ``xy_positions`` must match ``sym.block_names`` (from
    :func:`infer_block_layout` / the same program used to build ``sym``): sorted
    unique names for ByName programs, or ``str(int_id)`` for id-only programs.
    """
    hx, hy, hz = ee_home
    cons: List[z3.BoolRef] = [
        sym.ee_x[0] == hx,
        sym.ee_y[0] == hy,
        sym.ee_z[0] == hz,
        sym.holding[0] == sym.NONE,
    ]
    for b in sym.block_names:
        if b not in xy_positions:
            raise KeyError(f"default_table_initial_state: missing xy for block {b!r}")
        x0, y0 = xy_positions[b]
        cons.append(sym.bx[b][0] == x0)
        cons.append(sym.by[b][0] == y0)
        cons.append(sym.bz[b][0] == table_z)
    return cons
