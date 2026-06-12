import os
from copy import deepcopy
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

from synthesis.util import on


class Feature:
    pass


class ON_feature(Feature):
    """Represent the on(b1, b2) feature"""

    def __init__(self, b1: int, b2: int):
        self.b1 = b1
        self.b2 = b2

    def __call__(self, obs):
        return on.on(on.get_block_pos(obs, self.b1), on.get_block_pos(obs, self.b2))

    def __str__(self) -> str:
        return f"ON({self.b1}, {self.b2})"

    def __repr__(self) -> str:
        return self.__str__()


def compute_ON_features(num_block: int):
    features = []
    for b1 in range(num_block):
        for b2 in range(num_block):
            features.append(ON_feature(b1, b2))
    return features


def learn_features(
    num_block: int, sample_trajs, demos, demo_goal_idxs, features, num_trees: int
):
    """learn the feature that could seperate the positive and negative samples"""
    positive_samples = compute_positive_set(demos, demo_goal_idxs)
    negative_samples = compute_negative_set(sample_trajs)
    all_samples, all_features, all_labels = compute_features(
        positive_samples, negative_samples, features
    )
    all_trees = []
    for i in range(num_trees):
        clf = tree.DecisionTreeClassifier(max_depth=1, random_state=i)
        clf = clf.fit(all_features, all_labels)
        training_score = clf.score(all_features, all_labels)
        all_trees.append((clf, training_score))

    best_feature_id, best_tree, best_goal_idx = None, None, None
    for idx, (cur_tree, training_score) in enumerate(all_trees):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        tree.plot_tree(cur_tree, filled=True, rounded=True, ax=axes)
        plt.savefig(f"decision_tree{idx}.png", dpi=300, bbox_inches="tight")

        if training_score < 1.0:
            continue
        # for each classifying feature, check the first index that this feature becomes true after
        feature_id = cur_tree.tree_.feature[0]
        avg_goal_idx, _ = check_feature(features[feature_id], demos, demo_goal_idxs)
        print(
            f"learned tree {idx} is using feature {features[feature_id]} with average goal index {avg_goal_idx}"
        )
        if best_goal_idx is None or best_goal_idx > avg_goal_idx:
            best_feature_id, best_goal_idx, best_tree = (
                feature_id,
                avg_goal_idx,
                cur_tree,
            )

    if best_feature_id is None:
        print("No feature perfectly separates positive and negative samples")
        return None, None

    print(
        f"selecting tree with feature {features[best_feature_id]} with average goal idx {best_goal_idx}"
    )
    return best_tree, features[best_feature_id]


def find_first_feature_true_index(feature, demo) -> Optional[int]:
    """Return the first timestep index where ``feature`` is true."""
    for idx, state in enumerate(demo):
        if feature(state):
            return idx
    return None


def split_demo_at_feature(feature, demo) -> tuple[list, list, Optional[int]]:
    """Split one demo into (part1, part2, split_idx).

    Part 1 runs from the start through the first state where the feature is
    true (inclusive). Part 2 contains all later states.
    """
    split_idx = find_first_feature_true_index(feature, demo)
    if split_idx is None:
        return list(demo), [], None
    return demo[: split_idx + 1], demo[split_idx + 1 :], split_idx


def _split_frames(frames: list, split_idx: Optional[int]) -> tuple[list, list]:
    if split_idx is None:
        return frames, []
    return frames[: split_idx + 1], frames[split_idx + 1 :]


def split_demos_by_feature(
    feature,
    demos: list,
    *,
    demo_imgs: Optional[list[list]] = None,
    save_videos: bool = False,
    video_dir: str = "demo_splits",
    video_fps: int = 30,
) -> list[dict[str, Any]]:
    """Split each demo at the first state where ``feature`` becomes true.

    Returns one dict per demo with keys ``part1``, ``part2``, ``split_idx``,
    and ``demo_idx``. When ``save_videos`` is True, writes ``part1.mp4`` and
    ``part2.mp4`` under ``video_dir/demo_XXXX/``. ``demo_imgs`` must be a
    per-demo list of frame sequences aligned with ``demos``.
    """
    if save_videos and demo_imgs is None:
        raise ValueError("demo_imgs is required when save_videos=True")
    if demo_imgs is not None and len(demo_imgs) != len(demos):
        raise ValueError("demo_imgs must have the same length as demos")

    synthesis_mod = None
    if save_videos:
        from synthesis.mcmc import synthesis as synthesis_mod

    results: list[dict[str, Any]] = []
    for demo_idx, demo in enumerate(demos):
        part1, part2, split_idx = split_demo_at_feature(feature, demo)
        entry: dict[str, Any] = {
            "demo_idx": demo_idx,
            "split_idx": split_idx,
            "part1": part1,
            "part2": part2,
        }
        if split_idx is None:
            print(
                f"demo {demo_idx}: feature {feature} never became true; "
                "keeping full trajectory in part1"
            )
        else:
            print(
                f"demo {demo_idx}: split at index {split_idx} "
                f"({len(part1)} + {len(part2)} states)"
            )

        if save_videos:
            frames = demo_imgs[demo_idx]
            if len(frames) != len(demo):
                raise ValueError(
                    f"demo {demo_idx}: expected {len(demo)} frames, got {len(frames)}"
                )
            part1_frames, part2_frames = _split_frames(frames, split_idx)
            demo_video_dir = os.path.join(video_dir, f"demo_{demo_idx:04d}")
            synthesis_mod.save_frames_as_video(
                part1_frames,
                os.path.join(demo_video_dir, "part1.mp4"),
                fps=video_fps,
            )
            synthesis_mod.save_frames_as_video(
                part2_frames,
                os.path.join(demo_video_dir, "part2.mp4"),
                fps=video_fps,
            )
            entry["part1_video"] = os.path.join(demo_video_dir, "part1.mp4")
            entry["part2_video"] = os.path.join(demo_video_dir, "part2.mp4")

        results.append(entry)
    return results


def check_feature(feature, demos, demo_goal_idxs):
    """return the average index that the feature becomes true afterwards"""
    first_idxs = []
    for demo, goal_idx in zip(demos, demo_goal_idxs):
        assert goal_idx > 0
        assert feature(demo[goal_idx - 1])
        cur_idx = goal_idx
        while cur_idx > 0:
            if not feature(demo[cur_idx - 1]):
                first_idxs.append(cur_idx)
                break
            cur_idx -= 1
    return sum(first_idxs) / len(first_idxs), first_idxs


def compute_features(positive_samples, negative_samples, features):
    all_samples = positive_samples + negative_samples
    all_labels = [1 for _ in range(len(positive_samples))] + [
        0 for _ in range(len(negative_samples))
    ]
    all_features = []
    for sample in all_samples:
        all_features.append([f(sample) for f in features])
    return all_samples, all_features, all_labels


def compute_negative_set(sample_trajs):
    """sample_trajs is a 2D list of sampled trajectories using current program"""
    negative_samples = []
    for traj in sample_trajs:
        for state in traj:
            negative_samples.append(deepcopy(state))
    return negative_samples


def compute_positive_set(demos, demo_goal_idxs):
    """demos is 2D array of demo trajectories, demo_gaol_idxs is a list containing current idxs"""
    positive_samples = []
    assert len(demos) == len(demo_goal_idxs)
    for demo, goal_idx in zip(demos, demo_goal_idxs):
        if goal_idx == 0:
            print("current goal idx is already 0, aborting")
            exit()
        positive_samples.append(deepcopy(demo[goal_idx - 1]))
    return positive_samples
