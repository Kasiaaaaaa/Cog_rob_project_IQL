""" from rlenv import PegInHoleGymEnv

if __name__ == "__main__":
    shape = "circle"      # <-- put a valid value your env expects
    reward = "dense"      # <-- put a valid value your env expects

    env = PegInHoleGymEnv(shape_type=shape, reward_typ=reward)

    print("obs space:", env.observation_space)
    print("act space:", env.action_space)

    out = env.reset()
    # Gym 0.26+: reset returns (obs, info). Older gym: just obs.
    obs = out[0] if isinstance(out, tuple) else out
    print("reset obs shape:", getattr(obs, "shape", type(obs)))

    print("OK")
 """
#obs space: Dict('cam_image': Box(0, 255, (100, 100, 1), uint8))
#act space: Box(-0.005, 0.005, (3,), float32)

""" 
import numpy as np
d = np.load("peg_in_hole_iql_dataset.npz", allow_pickle=True)

for k in d.files:
    arr = d[k]
    print(k, arr.shape, arr.dtype) """

import numpy as np
import gym

from rlenv import PegInHoleGymEnv

# --- load dataset ---
data = np.load("iql_dataset_clean.npz")
actions = data["actions"]

print("Dataset actions:")
print("  min:", actions.min(axis=0))
print("  max:", actions.max(axis=0))
print("  overall min:", actions.min())
print("  overall max:", actions.max())

# --- load env ---
env = PegInHoleGymEnv(shape_type="circle", reward_typ="old")

print("\nEnvironment action space:")
print("  low:", env.action_space.low)
print("  high:", env.action_space.high)
