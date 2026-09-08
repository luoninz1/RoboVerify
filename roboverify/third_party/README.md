# Legacy Gym source

The Fetch construction environment imports `gym.envs.robotics`, removed in newer Gym releases.
This is the official Gym 0.21.0 source distribution, with only the malformed optional
`opencv-python>=3.` requirement corrected to `opencv-python>=3.0` in packaging metadata.
Environment code and the included MIT license are unchanged.

Upstream: https://files.pythonhosted.org/packages/4b/48/920cea66177b865663fde5a9390a59de0ef3b642ad98106ac1d8717d7005/gym-0.21.0.tar.gz

Upstream SHA-256: `0fd1ce165c754b4017e37a617b097c032b8c3feb8a0394ccc8777c7c50dddff3`

Patched files: gym-0.21.0/gym.egg-info/requires.txt, gym-0.21.0/setup.py.
