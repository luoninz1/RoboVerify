import itertools

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from sympy import And, Equivalent, Implies, Not, Or, symbols, to_cnf
from synthesis.inference_lib import inference


def generate_truth_table(expression, var_names):
    """
    Generates and prints a truth table for a given SymPy boolean expression.

    Args:
        expression: The SymPy boolean expression.
        var_names: A list of strings with the names of the variables.
    """
    # Define symbols
    syms = symbols(var_names)
    if isinstance(syms, str):  # Handle single variable case
        syms = [syms]

    # Header
    header = " | ".join(var_names) + " || Result"
    separator = "-" * len(header)
    print(header)
    print(separator)

    # Generate all combinations of True/False
    for combo in itertools.product([True, False], repeat=len(syms)):
        # Create a dictionary mapping symbols to their current values
        model = dict(zip(syms, combo))

        # Evaluate the expression using subs
        result = expression.subs(model)

        # Print the row
        row = " | ".join(f"{val!s:^5}" for val in combo) + f" || {result!s:^6}"
        print(row)


# Example Usage:
# Define variables
# A, B, C = symbols('A, B, C')
# # Define a boolean expression: (A and B) or C
# expr = Or(And(A, B), C)

# print("Truth Table for (A & B) | C:")
# generate_truth_table(expr, ['A', 'B', 'C'])

if __name__ == "__main__":
    import argparse

    from synthesis.entry.inference_options import add_inference_options
    from synthesis.inference_lib.demo_store import (
        DemoStore,
        InvInference,
        tower_vocabulary,
    )

    parser = argparse.ArgumentParser(
        description="Infer a tower invariant from loop-head traces."
    )
    parser.add_argument("--demos", required=True)
    parser.add_argument("--loop-id", default="1")
    add_inference_options(parser)
    parser.add_argument(
        "--task", choices=["stack", "unstack", "reverse", "partial"], default="stack"
    )
    args = parser.parse_args()
    context = highlevel_verification_lib.HighLevelContext(
        mode="declare",
        use_tbl=args.task in {"unstack", "reverse"},
        exists_top=args.task in {"unstack", "reverse"},
    )
    inferred_invariant, candidate_lists = InvInference(
        DemoStore.from_archive(args.demos),
        args.loop_id,
        tower_vocabulary(args.task),
        context,
        minimizer=args.invariant_minimizer,
    )
