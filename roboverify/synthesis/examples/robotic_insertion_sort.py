"""Physical insertion sort for Fetch construction's table blocks.

Only initial scene setup writes object poses. Pick, move, and release advance
the MuJoCo simulation through env.step; object motion comes from contact.
"""

from pathlib import Path
import shutil
import subprocess

import numpy as np


def block_position(env, name):
    return env.sim.data.get_site_xpos(name).copy()


def insertion_sort_blocks(order, keys, slots, buffer_position, robot):
    """Stable insertion sort with a physical buffer for the selected block.

    `order` is the mutable slot-to-block mapping. A None marks the empty slot.
    `robot` supplies transfer(name, target, description) and check_slots(order).
    """
    for i in range(1, len(order)):
        selected = order[i]
        if keys[order[i - 1]] <= keys[selected]:
            continue
        robot.transfer(selected, buffer_position, f"i={i}: save key {keys[selected]} in buffer")
        order[i] = None
        robot.check_slots(order)
        j = i - 1
        while j >= 0 and keys[order[j]] > keys[selected]:
            shifted = order[j]
            robot.transfer(shifted, slots[j + 1], f"i={i}: shift key {keys[shifted]} from slot {j} to {j + 1}")
            order[j + 1] = shifted
            order[j] = None
            robot.check_slots(order)
            j -= 1
        robot.transfer(selected, slots[j + 1], f"i={i}: insert key {keys[selected]} into slot {j + 1}")
        order[j + 1] = selected
        robot.check_slots(order)
        assert all(keys[order[k]] <= keys[order[k + 1]] for k in range(i))
    return order


class TableRobot:
    """Bounded, feedback-controlled pick/move/release primitives."""

    def __init__(self, env, slots, keys, recorder=None):
        self.env = env
        self.slots = np.asarray(slots)
        self.keys = keys
        self.recorder = recorder
        self.carry_height = env.height_offset + 0.19
        self.held = None
        self.grasp_offset = None
        self.steps = 0
        self.transfers = []
        self.description = "Initial shuffled row"
        self.phase = "Ready"

    @property
    def grip(self):
        return self.env.sim.data.get_site_xpos("robot0:grip").copy()

    def _step(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=float))
        self.steps += 1
        if not np.isfinite(obs).all():
            raise RuntimeError(f"Non-finite observation during {self.description}: {self.phase}")
        if self.recorder is not None:
            self.recorder.capture(self)

    def _hold(self, fingers, count):
        for _ in range(count):
            self._step([0, 0, 0, fingers])

    def _goto(self, target, fingers, tolerance=0.003, max_steps=160):
        target = np.asarray(target, dtype=float)
        for _ in range(max_steps):
            error = target - self.grip
            if np.linalg.norm(error) < tolerance:
                return
            # Up to 1.75 cm target displacement per simulation step.
            self._step(np.r_[np.clip(12.0 * error, -0.35, 0.35), fingers])
        raise RuntimeError(f"Gripper failed to reach {target.round(3)} during {self.phase}; actual={self.grip.round(3)}")

    def pick(self, name):
        if self.held is not None:
            raise RuntimeError("Cannot pick another block while holding one.")
        self.phase = f"PICK key {self.keys[name]}: approach"
        self._goto([*self.grip[:2], self.carry_height], 0.2)
        self._hold(0.2, 8)
        p = block_position(self.env, name)
        self._goto([*p[:2], self.carry_height], 0.2)
        self.phase = f"PICK key {self.keys[name]}: descend and close"
        self._goto(p, 0.2)
        self._hold(-0.2, 16)
        self.phase = f"PICK key {self.keys[name]}: lift"
        self._goto([*self.grip[:2], self.carry_height], -0.2)
        self._hold(-0.2, 5)
        lifted = block_position(self.env, name)
        if lifted[2] < self.env.height_offset + 0.10:
            raise RuntimeError(f"Grasp failed for {name}: block at {lifted}, gripper at {self.grip}")
        self.held = name
        self.grasp_offset = lifted - self.grip

    def move(self, target):
        if self.held is None:
            raise RuntimeError("move() requires a picked block.")
        self.phase = f"MOVE key {self.keys[self.held]} above destination"
        target = np.asarray(target)
        self._goto([*(target[:2] - self.grasp_offset[:2]), self.carry_height], -0.2)
        actual = block_position(self.env, self.held)
        if actual[2] < self.env.height_offset + 0.08:
            raise RuntimeError(f"Dropped {self.held} while carrying it.")

    def release(self, target):
        if self.held is None:
            raise RuntimeError("release() requires a picked block.")
        name = self.held
        target = np.asarray(target)
        self.phase = f"RELEASE key {self.keys[name]}: lower"
        # Bring the held block just above the table, then let contacts settle it.
        self._goto(target + [0, 0, 0.003] - self.grasp_offset, -0.2)
        self.phase = f"RELEASE key {self.keys[name]}: open and retreat"
        self._hold(0.2, 14)
        self.held = None
        self._goto([*self.grip[:2], self.carry_height], 0.2)
        self._hold(0, 10)
        actual = block_position(self.env, name)
        if np.linalg.norm(actual[:2] - target[:2]) > 0.012 or abs(actual[2] - target[2]) > 0.005:
            raise RuntimeError(f"Placement failed for {name}: expected {target}, got {actual}")
        return actual

    def transfer(self, name, target, description):
        self.description = description
        print(description, flush=True)
        start_step = self.steps
        self.pick(name)
        self.move(target)
        actual = self.release(target)
        self.transfers.append({"block": name, "key": self.keys[name], "description": description,
                               "target": np.asarray(target).tolist(), "actual": actual.tolist(),
                               "start_step": start_step, "end_step": self.steps})

    def check_slots(self, order):
        for i, name in enumerate(order):
            if name is not None:
                actual = block_position(self.env, name)
                if (np.linalg.norm(actual[:2] - self.slots[i, :2]) > 0.012
                        or abs(actual[2] - self.slots[i, 2]) > 0.005):
                    raise RuntimeError(f"Unexpected displacement of {name} from slot {i}: {actual}")


class SortVideo:
    """Stream labeled simulator frames to ffmpeg without storing a large list."""

    def __init__(self, env, keys, slots, buffer_position, path, width=960, height=720, stride=2):
        from PIL import ImageFont
        from matplotlib import font_manager
        self.env, self.keys, self.slots = env, keys, np.asarray(slots)
        self.buffer_position = np.asarray(buffer_position)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height, self.stride = width, height, stride
        self.fps = 1.0 / (env.sim.nsubsteps * env.sim.model.opt.timestep * stride)
        self.frames = 0
        font = font_manager.findfont("DejaVu Sans")
        self.font = ImageFont.truetype(font, 19)
        self.small = ImageFont.truetype(font, 16)
        ffmpeg = shutil.which("ffmpeg") or next((p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if Path(p).is_file()), None)
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to save the sorting video.")
        self.process = subprocess.Popen([
            ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(self.fps), "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def frame(self, robot):
        from PIL import Image, ImageDraw
        pixels = self.env.render(mode="rgb_array", width=self.width, height=self.height)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, 102), fill="#122337")
        draw.text((20, 9), "ROBOTIC INSERTION SORT", font=self.font, fill="white")
        for i, (name, key) in enumerate(self.keys.items()):
            model = self.env.sim.model
            geom_id = model.geom_name2id(name)
            material_id = model.geom_matid[geom_id]
            rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geom_id]
            color = tuple((255 * rgba[:3]).astype(int))
            x = 365 + 140 * i
            draw.rectangle((x, 12, x + 16, 28), fill=color)
            draw.text((x + 23, 9), f"key {key}", font=self.small, fill="white")
        order = []
        for slot in self.slots:
            occupant = next((str(self.keys[name]) for name in self.keys
                             if np.linalg.norm(block_position(self.env, name) - slot) < 0.025), "_")
            order.append(occupant)
        buffer_key = next((str(self.keys[name]) for name in self.keys
                           if np.linalg.norm(block_position(self.env, name) - self.buffer_position) < 0.025), "_")
        draw.text((20, 44), f"Row (increasing y):  [ {'  '.join(order)} ]     Buffer: {buffer_key}", font=self.font, fill="#e1efff")
        draw.text((20, 76), f"Simulation time: {robot.steps * self.env.dt:.1f} s", font=self.small, fill="#b5cbe3")
        draw.rectangle((0, self.height - 76, self.width, self.height), fill="#122337")
        draw.text((20, self.height - 67), robot.description, font=self.font, fill="white")
        draw.text((20, self.height - 34), robot.phase, font=self.small, fill="#b5cbe3")
        return np.asarray(image)

    def capture(self, robot, force=False):
        if force or robot.steps % self.stride == 0:
            self.process.stdin.write(self.frame(robot).tobytes())
            self.frames += 1

    def pause(self, robot, seconds=1):
        frame = self.frame(robot).tobytes()
        for _ in range(round(seconds * self.fps)):
            self.process.stdin.write(frame)
            self.frames += 1

    def close(self):
        if self.process.stdin.closed:
            return
        self.process.stdin.close()
        errors = self.process.stderr.read().decode()
        status = self.process.wait()
        self.process.stderr.close()
        if status:
            raise RuntimeError(f"Video encoding failed: {errors}")
