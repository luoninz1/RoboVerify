from __future__ import annotations

import numpy as np
import z3

BLOCK_LENGTH = (
    0.025 * 2
)  # (0.025, 0.025, 0.025) in the xml file of the gym env is half length


def on_star_eval(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def on(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2 < 1.5 * BLOCK_LENGTH
    )


def z3_on(
    x1: z3.ArithRef,
    y1: z3.ArithRef,
    z1: z3.ArithRef,
    x2: z3.ArithRef,
    y2: z3.ArithRef,
    z2: z3.ArithRef,
    block_length: float = BLOCK_LENGTH,
) -> z3.BoolRef:
    """Z3 encoding aligned with :func:`on` (same xy tolerance and vertical gap)."""
    half_xy = z3.RealVal(block_length / 2.0)
    max_dz = z3.RealVal(1.5 * block_length)
    return z3.And(
        z3.Abs(x1 - x2) < half_xy,
        z3.Abs(y1 - y2) < half_xy,
        z1 - z2 >= 0,
        z1 - z2 < max_dz,
    )


def on_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def d_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the d_star(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return x1 == x2 and z1 == z2 and y1 <= y2


def r_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the r_star(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return y1 == y2 and z1 == z2 and x1 <= x2


def higher_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (0 <= z1 - z2 and z1 >= 0.0 and z2 >= 0.0) or (
        x1 == x2 and y1 == y2 and z1 == z2
    )


def scattered_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the scattered(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return abs(x1 - x2) >= 2 * BLOCK_LENGTH or abs(y1 - y2) >= 2 * BLOCK_LENGTH


def top_implementation(block, all_blocks) -> bool:
    """Check if a block is on top"""
    top_flag = True
    for other_block in all_blocks:
        if other_block != block and on_star_implementation(other_block, block):
            top_flag = False
            break
    return top_flag


def get_block_pos(obs, block_id):
    start_idx = 10 + 12 * block_id
    end_idx = start_idx + 3
    return np.array(obs[start_idx:end_idx])


def print_block_layout(obs, num_block):
    for i in range(0, num_block):
        for j in range(0, num_block):
            print(f"on({i}, {j})", on(get_block_pos(obs, i), get_block_pos(obs, j)))
