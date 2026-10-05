# Fetch N-Push and N-Switch environments

These are upstream examples retained with the vendored environment. They assume
the upstream `envs` package layout; that import path and the corresponding Gym
registration entry points are not the RoboVerify package layout. They are not
commands for the supported Stack workflow. See the
[RoboVerify collection guide](../../inference_lib/README.md) for current usage.

Upstream initialization examples:

```python
import envs
import gym

env = gym.make("Fetch3Push-v1")  # 3-Push, sparse reward
env = gym.make("Fetch3PushDense-v1")  # 3-Push, dense reward

env = gym.make("Fetch3Switch-v1")  # 3-Switch, sparse reward

env = gym.make("Fetch2Switch2Push-v1")  # 2 switches and 2 cubes

env = gym.make("Fetch2SwitchOr2Push-v1")  # 2 switches and 2 cubes, but the goal only involves either the switches or the cubes
```
