from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):  # gymnasium: (obs, info)
            observation, _ = reset_out
        else:
            observation = reset_out

        done = False

        while not done:
            # Dict obs -> extract image
            if isinstance(observation, dict):
                obs_img = observation["cam_image"]
            else:
                obs_img = observation

            # agent expects batched obs, returns batched actions -> take [0]
            action = agent.sample_actions(obs_img[None, ...], temperature=0.0)[0]

            # scale action from [-1, 1] to env action space bounds
            low, high = env.action_space.low, env.action_space.high
            action = low + (action + 1.0) * 0.5 * (high - low)

            step_out = env.step(action)

            # gymnasium: 5-tuple, gym: 4-tuple
            if len(step_out) == 5:
                observation, _, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                observation, _, done, info = step_out
                done = bool(done)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = float(np.mean(v))

    return stats
