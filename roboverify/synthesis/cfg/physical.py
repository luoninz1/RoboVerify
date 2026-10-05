"""Lift numeric motion operands into demonstrated symbolic aliases, without a summary."""

from copy import deepcopy

from synthesis.api.instructions import (
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    Skip,
)


def name_operands(instruction, aliases):
    """aliases maps concrete object IDs to names available at this block entry."""
    if isinstance(instruction, Pick):
        return PickByName(
            aliases[instruction.grab_box_id],
            limit=instruction.limit,
            control=instruction.control,
        )
    if isinstance(instruction, Move):
        result = MoveByName(
            aliases[instruction.target_box_id_x],
            aliases[instruction.target_box_id_y],
            aliases[instruction.target_box_id_z],
            limit=instruction.limit,
            control=instruction.control,
        )
        result.target_offset = deepcopy(instruction.target_offset)
        return result
    if isinstance(instruction, Release):
        result = ReleaseByName(
            aliases[instruction.release_box_id],
            limit=instruction.limit,
            control=instruction.control,
        )
        result.target_z_offset = deepcopy(instruction.target_z_offset)
        return result
    return deepcopy(instruction)
