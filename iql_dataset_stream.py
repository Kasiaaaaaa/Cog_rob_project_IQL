import numpy as np
import pickle
from stable_baselines3 import SAC

# -------- CONFIG --------
MODEL_PATH = "sac_model.zip"
REPLAY_PATH = "replay_buffer.pkl"
OUT_PATH = "iql_dataset_full.npz"
CHUNK_SIZE = 100_000        # tune: 50k–200k safe on most machines
# ------------------------

print("Loading model (no env)...")
model = SAC.load(MODEL_PATH, env=None)
model.load_replay_buffer(REPLAY_PATH)

rb = model.replay_buffer
N = rb.size()

print(f"Replay buffer size: {N}")

# Infer shapes
obs_shape = rb.observations["cam_image"].shape[1:]
act_shape = rb.actions.shape[1:]

# Create memory-mapped arrays on disk (NOT RAM)
obs_mm = np.memmap("obs.dat", dtype=np.float32, mode="w+", shape=(N, *obs_shape))
next_obs_mm = np.memmap("next_obs.dat", dtype=np.float32, mode="w+", shape=(N, *obs_shape))
act_mm = np.memmap("actions.dat", dtype=np.float32, mode="w+", shape=(N, *act_shape))
rew_mm = np.memmap("rewards.dat", dtype=np.float32, mode="w+", shape=(N,))
done_mm = np.memmap("terminals.dat", dtype=np.float32, mode="w+", shape=(N,))

print("Streaming replay buffer → disk...")

for start in range(0, N, CHUNK_SIZE):
    end = min(start + CHUNK_SIZE, N)
    print(f"  [{start}:{end}]")

    obs_mm[start:end] = rb.observations["cam_image"][start:end].astype(np.float32)
    next_obs_mm[start:end] = rb.next_observations["cam_image"][start:end].astype(np.float32)
    act_mm[start:end] = rb.actions[start:end].astype(np.float32)
    rew_mm[start:end] = rb.rewards[start:end].reshape(-1)
    done_mm[start:end] = rb.dones[start:end].reshape(-1)

# Flush to disk
del obs_mm, next_obs_mm, act_mm, rew_mm, done_mm

print("Packing into NPZ (final step)...")

np.savez(
    OUT_PATH,
    observations=np.memmap("obs.dat", dtype=np.float32, mode="r", shape=(N, *obs_shape)),
    next_observations=np.memmap("next_obs.dat", dtype=np.float32, mode="r", shape=(N, *obs_shape)),
    actions=np.memmap("actions.dat", dtype=np.float32, mode="r", shape=(N, *act_shape)),
    rewards=np.memmap("rewards.dat", dtype=np.float32, mode="r", shape=(N,)),
    terminals=np.memmap("terminals.dat", dtype=np.float32, mode="r", shape=(N,))
)

print("Done ✅:", OUT_PATH)
