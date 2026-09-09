# Robotic insertion sort

- [x] Inspect Fetch action API and choose a temporary table position for insertion sort's saved block.
- [x] Assign stable block keys and initialize a reproducibly shuffled row.
- [x] Execute insertion sort with physical pick, move, and release actions.
- [x] Verify grasping, placements, final physical order, and object preservation.
- [x] Execute the notebook, inspect the MP4, and record validation evidence.

## Verified results (2026-09-08)

The executed notebook and `robotic_insertion_sort.mp4` show seed 42 sorting
`[4, 3, 2, 1]` to `[1, 2, 3, 4]`. Twelve transfers use 1,975 physics steps
(79 simulated seconds). Final maximum 3D slot error is 2.49 mm.
The four block identities are preserved, each occupies the expected row slot,
each remains upright and at tabletop height, and the buffer and gripper are empty.
Transfer destinations and measured results are in `robotic_insertion_sort_result.json`.

The MP4 is H.264, 960×720, 12.5 fps, 1,050 frames, 84 seconds including pauses.
Full ffmpeg decoding passed without errors. Six intermediate frames and the final
image were visually checked: the robot grasps/lifts/carries/releases the blocks,
the buffer is used, keys match the block colors, and the final row is sorted.
The video is also embedded in the executed notebook.

Additional headless MuJoCo checks in `robotic_insertion_sort_validation.json`:

| Seed | Initial keys | Final keys | Transfers | Steps | Maximum XY slot error |
| --- | --- | --- | --- | --- | --- |
| 7 | 1, 3, 2, 4 | 1, 2, 3, 4 | 3 | 495 | 2.11 mm |
| 21 | 1, 2, 4, 3 | 1, 2, 3, 4 | 3 | 502 | 1.69 mm |

Four algorithm tests passed, including all 24 unique-key permutations, all 24
identity permutations with duplicate keys (stable ordering), physical occupancy
constraints, and empty/singleton/already-sorted cases. These use a fake occupancy
model and do not establish MuJoCo success for all permutations.

```bash
unset LD_PRELOAD
uv run python3 -m unittest synthesis.examples.test_robotic_insertion_sort -v
uv run python3 -m jupyter nbconvert --to notebook --execute notebooks/robotic_insertion_sort.ipynb --inplace --ExecutePreprocessor.timeout=600
```

For the moving robot, slot spacing was increased from 9 to 11 cm after a live
9 cm test detected a neighboring block displaced by an open gripper. Blocks
remain 5 cm cubes. Scene restoration now includes actuator inputs, mocap targets,
applied forces, and solver warm-start values in addition to legacy GT state.

Scope: four simulated Fetch table cubes, unique keys 1–4. The simulator's measured
poses provide feedback. The simulation demonstration does not establish physical
robot deployment or formal correctness of the controller.
