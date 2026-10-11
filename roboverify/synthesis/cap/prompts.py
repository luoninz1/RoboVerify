"""Fixed API descriptions and examples; benchmark solutions are not examples."""

from synthesis.benchmarks.specs import DESCRIPTIONS, ROW_TASKS

COMMON = """
Write executable Python functions for a robot. Return code only.
Define def policy(): with no arguments. It must solve the task for any valid
scene and any provided number of objects, not just one example.
You may define helpers or call missing helpers that will be generated later.
Allowed Python: def, local variables, if, for, while, break, continue, return,
assert, list/tuple/dict, comprehensions, ordinary arithmetic/comparisons.
Builtins: all, any, len, range, enumerate, list, tuple, reversed, zip, next,
min, max, abs, bool, int, float, dict, ValueError, RuntimeError.
List methods: append, extend, pop, index, remove, copy, reverse.
Dict methods: get, keys, values, items, copy. Tuple: index, count.
No imports, numpy, classes, lambda, try/except, raw env, raw coordinates,
image access, files, exec/eval, or direct simulator access.
Do not reuse an API name as a variable, parameter name, or helper name.
objects() returns a fresh list of all object handles in arbitrary identity
order, unrelated to spatial order or key order. == and != compare identities.
parameter(name, default, low, high) returns a scalar, initially default, that
may be tuned later within the stated bounds. Repeated use of a name must have
the same default and bounds. Use this for motion offsets worth tuning. There
is no reward query or goal checker exposed to the policy.

"""

TOWER = """
Tower API (distances in metres; upright cubes have side length 0.05):
pick(obj): approach horizontally at current gripper height, descend to obj,
then close. The gripper must be empty. Use objects with no other object above.
move(x_ref, y_ref, z_ref, offset=(dx,dy,dz)): move the gripper to the CURRENT
position [x(x_ref)+dx, y(y_ref)+dy, z(z_ref)+dz]. Reference handles can differ
by axis. If a referenced object is held, its current position also changes
after each move. The instruction does not auto-lift or plan around obstacles.
release(obj, retreat_offset=0.15): open at the current position, then retreat
vertically by retreat_offset relative to obj. Name the held object. This is
NOT a placement-height argument. Lower to support with move BEFORE release.
All controller gains, tolerances, and 50-step budgets are fixed across methods.
Transport should first lift vertically, then translate above all obstacles,
then descend vertically to the placement. Picking below another block is
unsafe. Placing on a block requires a 0.05 height difference.
on_star(a,b): same XY column and a at least as high as b; reflexive.
higher(a,b): z(a) >= z(b) - 0.001; reflexive. It does not imply ON_star.
scattered(a,b): abs(dx)>=0.10 OR abs(dy)>=0.10, regardless of heights.
on_star_zero(a,b): frozen initial ON_star, unaffected by later moves.
base is a designated input object. Additional roles root_a, root_b, cut are
supplied only when they are part of the task. They are ordinary object handles.
Input scenes are legal, supported towers or separated table blocks. Initial
gripper height clears all input objects. Empty table destinations must be
chosen using the same move API, e.g. offsets from an unmoved reference.

Example of a predicate helper (not a task solution):
def has_other_above(obj):
    return any(other != obj and on_star(other, obj) for other in objects())

Example of numeric parameter syntax:
clearance = parameter("clearance", 0.15, 0.10, 0.25)
Object identities such as base are supplied inputs, NEVER numeric parameters.
"""

ROW = """
Row API: objects have hidden integer keys and occupy a row of fixed table slots.
next_star(a,b): a's current row position is before or equal to b's. Reflexive.
Available only when every slot is occupied and the buffer and gripper are empty.
keyless(a,b): key(a) < key(b). Keys never change. No numeric key values exposed.
target(obj): an opaque handle for obj's current row slot. Save this BEFORE
moving obj. The handle remains fixed after obj moves and cannot be inspected.
buffer_target(): an opaque handle for the single temporary buffer location.
pick(obj): physically pick and lift obj. Requires an empty gripper.
move(destination): carry above a previously saved target or buffer_target().
release(destination): lower, open, and retreat at that target. Requires a held
object. The destination must be empty; never place a block on an occupied slot.
All transfers use the existing row controller and its fixed physical settings.
The buffer holds at most one object. Return complete row permutations before
querying next_star again. Save every target needed by a transfer sequence before
moving its occupant. Local lists of handles and previously queried relations
are allowed. base is the object initially in the leftmost row slot.
Equal keys: neither keyless(a,b) nor keyless(b,a). Preserve their initial order.

Example of a relational helper (not a task solution):
def is_adjacent(a, b):
    return a != b and next_star(a, b) and all(
        obj == a or obj == b or not (next_star(a, obj) and next_star(obj, b))
        for obj in objects())
"""


def task_prompt(task):
    return (
        COMMON
        + (ROW if task in ROW_TASKS else TOWER)
        + "\nTask:\n"
        + DESCRIPTIONS[task]
    )
