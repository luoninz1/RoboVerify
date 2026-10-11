"""Adapters to existing RoboVerify checks with explicit coverage and scope.

Concrete primitive traces can be checked in the noiseless waypoint model. This
does not prove a Python policy's guards, all initial scenes, or controller
refinement. Row-controller paths are explicitly unsupported by this encoding.
"""

from dataclasses import asdict


def verify_tower_trace(execution, *, timeout_ms=1000):
    from synthesis.api.instructions import MoveByName, PickByName, ReleaseByName
    from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
    from synthesis.verification_lib.motion_verification import MotionContract
    from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem

    scope = "noiseless finite-scene primitive waypoint trace; no policy or controller-refinement proof"
    if "scene" not in execution:
        return {
            "status": "unsupported",
            "scope": scope,
            "reason": "No executable scene",
            "checks": [],
        }
    blocks = execution.get("motion_blocks", [])
    if not blocks:
        return {
            "status": "unsupported",
            "scope": scope,
            "reason": "No covered primitive motion blocks",
            "checks": [],
        }
    covered = sum(len(block["instructions"]) for block in blocks)
    if covered != execution.get("primitive_calls") or execution.get("error"):
        return {
            "status": "unsupported",
            "scope": scope,
            "reason": "Incomplete or failed primitive execution; prefix checks cannot certify the trace",
            "checks": [],
        }
    all_checks = []
    for index, block in enumerate(blocks):
        body = []
        for item in block["instructions"]:
            if item["kind"] == "pick":
                body.append(PickByName(item["object"]))
            elif item["kind"] == "move":
                body.append(
                    MoveByName(*item["references"], target_offset=item["offset"])
                )
            else:
                body.append(
                    ReleaseByName(item["object"], target_z=item["retreat_offset"])
                )
        # All actual obstacles are named. sym is fixed outside the finite scene;
        # existing named checks are retained. This does NOT cover unnamed blocks.
        positions = dict(block["positions"], sym=[100.0, 100.0, -100.0])
        context = LowLevelContext(default_L=0.05)
        contract = MotionContract(
            block["source"],
            block["source"],
            table_surface_height=execution["scene"]["table_center"] - 0.025,
        )
        problem = PrimitiveMotionProblem(
            context,
            [],
            list(block["positions"]),
            contract,
            None,
            str(index),
            timeout_ms,
            initial_positions=positions,
            initial_arm=block["arm"],
        )
        first = problem.check("initial_consistency")
        if first.status == "valid":
            problem.execute(body)
            problem.check("transition_consistency")
        for check in problem.checks:
            record = asdict(check)
            record["block"] = index
            all_checks.append(record)
        if any(check.status != "valid" for check in problem.checks):
            break
    failed = [check for check in all_checks if check["status"] != "valid"]
    return {
        "status": (
            "valid"
            if all_checks and not failed
            else failed[0]["status"] if failed else "unsupported"
        ),
        "scope": scope,
        "checked_blocks": index + 1,
        "checks": all_checks,
        "failures": failed,
    }
