import copy

import gym
import numpy as np
from gym.spaces import Box, Dict
import copy
import gym
import numpy as np
from gym.spaces import Box, Dict

def _is_image_box(space: Box) -> bool:
    # Heuristic: uint8 OR [0,255] bounds typical for images
    if space.dtype == np.uint8:
        return True
    try:
        low_min = np.min(space.low)
        high_max = np.max(space.high)
        if low_min >= 0 and high_max <= 255 and space.shape[-1] in (1, 3, 4):
            return True
    except Exception:
        pass
    return False

class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            if _is_image_box(obs_space):
                # Keep image space as-is (uint8)
                self.observation_space = Box(obs_space.low, obs_space.high,
                                             obs_space.shape, dtype=obs_space.dtype)
            else:
                # Cast continuous spaces to float32
                self.observation_space = Box(obs_space.low.astype(np.float32),
                                             obs_space.high.astype(np.float32),
                                             obs_space.shape, dtype=np.float32)

        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                if not isinstance(v, Box):
                    raise NotImplementedError(f"SinglePrecision Dict only supports Box, got {type(v)} for key {k}")
                if _is_image_box(v):
                    obs_spaces[k] = Box(v.low, v.high, v.shape, dtype=v.dtype)
                else:
                    obs_spaces[k] = Box(v.low.astype(np.float32),
                                        v.high.astype(np.float32),
                                        v.shape, dtype=np.float32)
            self.observation_space = Dict(obs_spaces)

        else:
            raise NotImplementedError(f"Unsupported observation space: {type(self.observation_space)}")

    def observation(self, observation):
        # Keep image observations uint8; cast only non-image to float32
        if isinstance(observation, np.ndarray):
            if observation.dtype == np.uint8:
                return observation
            return observation.astype(np.float32)

        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                if isinstance(v, np.ndarray) and v.dtype == np.uint8:
                    observation[k] = v
                else:
                    observation[k] = v.astype(np.float32)
            return observation

        return observation

""" 
class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = Box(obs_space.low, obs_space.high,
                                         obs_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
 """