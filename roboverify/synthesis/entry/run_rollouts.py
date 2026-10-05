import os
import shutil
from pathlib import Path
from typing import Any, Callable

import ffmpeg
import imageio
import numpy as np

from synthesis.api.program import Program


def run_program_rollouts(
    program: Program,
    *,
    env_factory: Callable[[int], Any],
    num_seeds: int,
    timesteps: int | None = None,
    return_img: bool = True,
    save_dir: str = "program_rollouts",
    fps: int = 20,
    save_png_frames: bool = True,
    video_filename: str = "000_trajectory.mp4",
    on_loop_head=None,
) -> dict:
    """Run `program` for multiple seeds and optionally save frames/video.

    `env_factory(seed)` should construct and return an environment instance.
    If `timesteps` is not None, we best-effort set `max_step` on the env.
    Pass ``on_loop_head=store.add`` to collect into a DemoStore across seeds;
    each loop invocation records its own entry geometry.
    """

    results: list[dict] = []
    save_root = Path(save_dir)
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    for seed in range(int(num_seeds)):
        np.random.seed(int(seed))
        env = env_factory(int(seed))

        if timesteps is not None:
            if hasattr(env, "max_step"):
                env.max_step = int(timesteps)
            if hasattr(env, "env") and hasattr(env.env, "max_step"):
                env.env.max_step = int(timesteps)

        seed_dir = save_root / f"seed_{seed:04d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        video_path = seed_dir / video_filename

        frame_idx = 0
        timed_out = False

        if return_img:
            traj, imgs = program.eval(env, return_img=True, on_loop_head=on_loop_head)
            for frame in imgs:
                frame_arr = np.asarray(frame, dtype=np.uint8)
                if save_png_frames:
                    imageio.imwrite(seed_dir / f"img{frame_idx:04d}.png", frame_arr)
                frame_idx += 1
        else:
            traj = program.eval(env, return_img=False, on_loop_head=on_loop_head)

        final_obs = traj[-1] if traj else None
        success = (
            bool(env.env._is_success(final_obs))
            if (final_obs is not None and hasattr(env, "env"))
            else False
        )

        if return_img and save_png_frames and frame_idx > 0:
            input_pattern = os.fspath(seed_dir / "img%04d.png")
            try:
                (
                    ffmpeg.input(input_pattern, framerate=int(fps))
                    .output(
                        os.fspath(video_path),
                        vcodec="libx264",
                        pix_fmt="yuv420p",
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
            except Exception as e:
                print(
                    "[WARN] failed to assemble mp4 with ffmpeg; PNG frames are still saved.",
                    repr(e),
                )

        results.append(
            {
                "seed": int(seed),
                "success": success,
                "timed_out": bool(timed_out),
                "dir": os.fspath(seed_dir),
                "frames": int(frame_idx),
            }
        )

        if hasattr(env, "close"):
            env.close()

    success_rate = sum(r["success"] for r in results) / max(1, len(results))
    return {"results": results, "success_rate": success_rate}
