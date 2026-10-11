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
min, max, abs, bool, int, float, str, dict, sorted, sum, ValueError, RuntimeError.
List methods: append, extend, pop, index, remove, copy, reverse, insert, sort, count.
Dict methods: get, keys, values, items, copy. Tuple: index, count.
Lambda functions are allowed for local computations or sorting keys. F-strings
are allowed for messages and parameter names, not hidden coordinate access.
No imports, numpy, classes, try/except, raw env, raw coordinates,
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

Example of composing primitives for ONE transfer, not a task algorithm:
def place_one_example(source, support):
    pick(source)
    move(source, source, source, (0.0, 0.0, 0.20))
    move(support, support, source, (0.0, 0.0, 0.0))
    move(support, support, support, (0.0, 0.0, 0.05))
    release(source, retreat_offset=0.15)
This example is suitable only if source is exposed, support is the intended
top support, and the lifted height clears intervening obstacles. It is not an
additional API. Numeric move(0,0,z) and z(obj) are NOT available. Use relation
queries instead of numeric positions. Sorting may use only permitted relations,
for example counts of related objects, never inaccessible XYZ values.
Use the supplied base handle for the requested bottom, not objects()[0].
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

Example of ONE transfer to an already empty destination handle:
def transfer_one_example(obj, destination):
    pick(obj)
    move(destination)
    release(destination)
This is composed from primitives, not an additional API. A row initially has
no empty row slot. The buffer can create a vacancy, but every displaced object
must eventually return to the row. A local list may track the row order while a
transfer sequence is incomplete. sorted() changes only a local list, NEVER the
physical row. Keys are opaque: sorting must use keyless comparisons or derived
relational ranks, not a hidden numeric-key accessor.

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
        + "\nInitial scene:\n"
        + INITIAL_SCENES[task]
        + "\nRequired final result:\n"
        + DESCRIPTIONS[task]
    )


INITIAL_SCENES = {
    "stack": "Every object initially rests separately on the table, pairwise Scattered. They are NOT already in a tower. The supplied base is the required final bottom object.",
    "unstack": "Every object initially belongs to one supported tower, with supplied base at its bottom. There is initially NO separate table block to use as a destination; choose empty table locations via offsets from an unmoved reference.",
    "reverse": "Every object initially belongs to one supported tower, with supplied base at its bottom. The final bottom must be the original top object, not necessarily base.",
    "partial_stack": "There are two separated supported towers. All objects must be combined. The supplied base is the required final bottom but may initially be inside one tower, with blocks above AND below it.",
    "tower_merge": "There are two separated supported towers, rooted at supplied root_a and root_b. The A tower must not move. Identify each tower's members and order through on_star_zero, not objects() order.",
    "tower_split": "There is one supported tower. Supplied cut is inside it, not necessarily its top. The portion strictly below cut must not move; cut and its initial successors must form a new, separate tower.",
    "insertion_sort": "Every fixed row slot is occupied, buffer empty. Initial key order is arbitrary and may contain duplicates or negative values. Recover spatial order with next_star, not objects() order.",
    "stable_merge": "Every fixed row slot is occupied, buffer empty. The left half (len(objects())//2 slots) and remaining right half are independently nondecreasing by key; duplicates are possible. Recover spatial order with next_star.",
    "sort_restore": "Every fixed row slot is occupied, buffer empty, initial key order arbitrary. Save the ORIGINAL object order through next_star before moving anything. Equal keys must retain their original order in the sorted checkpoint.",
}
