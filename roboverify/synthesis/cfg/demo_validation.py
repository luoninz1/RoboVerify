"""Task validation for recordings, independent of their source or controller."""

from dataclasses import dataclass, field

from synthesis.cfg.refine import scene_at
from synthesis.predicates.scene import evaluate


@dataclass
class DemoValidation:
    total: int
    pre_holds: int = 0
    post_at_end: int = 0
    post_reached: int = 0
    issues: list = field(default_factory=list)

    def __bool__(self):
        return self.total > 0 and not self.issues

    def as_dict(self):
        return dict(
            total=self.total,
            pre_holds=self.pre_holds,
            post_at_end=self.post_at_end,
            post_reached=self.post_reached,
            issues=self.issues,
        )


def validate_demonstrations(segments, pre, post):
    result = DemoValidation(len(segments))
    if not segments:
        result.issues.append({"demo": None, "reason": "No demonstrations supplied"})
    for segment in segments:
        try:
            initial = evaluate(pre, scene_at(segment, segment.t_start))
            final = evaluate(post, scene_at(segment, segment.t_end))
            reached = any(
                evaluate(post, scene_at(segment, t))
                for t in range(segment.t_start, segment.t_end + 1)
            )
        except (KeyError, ValueError, TypeError) as exc:
            result.issues.append({"demo": segment.demo_idx, "reason": str(exc)})
            continue
        result.pre_holds += int(initial)
        result.post_at_end += int(final)
        result.post_reached += int(reached)
        if not initial:
            result.issues.append(
                {"demo": segment.demo_idx, "reason": "Initial task condition is false"}
            )
        if not final:
            result.issues.append(
                {
                    "demo": segment.demo_idx,
                    "reason": "Final task condition is false",
                    "post_reached_transiently": reached,
                }
            )
    return result
