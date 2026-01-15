import os
import numpy as np
from stable_baselines3 import SAC

# ------------- CONFIG -------------
MODEL_PATH = "sac_model.zip"
REPLAY_PATH = "replay_buffer.pkl"

OUT_DIR = "./iql_dataset_npy"      # folder to write .npy files
CHUNK_SIZE = 100_000              # lower if you still get killed (e.g., 20_000)
STORE_OBS_UINT8 = True            # True saves disk; False writes float32
# ----------------------------------


def open_npy_memmap(path: str, shape, dtype):
    """
    Creates a .npy file as a memmap (writes header once, then allows chunk writes).
    """
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model (env=None)...")
    model = SAC.load(MODEL_PATH, env=None)
    print("Loading replay buffer...")
    model.load_replay_buffer(REPLAY_PATH)
    rb = model.replay_buffer

    N = rb.size()
    print(f"Replay buffer size: {N}")

    # SB3 DictReplayBuffer: observations is dict
    obs_src = rb.observations["cam_image"]
    next_obs_src = rb.next_observations["cam_image"]

    obs_shape = obs_src.shape[1:]         # e.g. (100,100,1)
    act_shape = rb.actions.shape[1:]      # e.g. (3,)

    # Choose observation dtype
    obs_dtype = np.uint8 if STORE_OBS_UINT8 else np.float32

    obs_path = os.path.join(OUT_DIR, "observations.npy")
    next_obs_path = os.path.join(OUT_DIR, "next_observations.npy")
    act_path = os.path.join(OUT_DIR, "actions.npy")
    rew_path = os.path.join(OUT_DIR, "rewards.npy")
    term_path = os.path.join(OUT_DIR, "terminals.npy")

    print("Creating .npy memmaps on disk...")
    obs_mm = open_npy_memmap(obs_path, (N, *obs_shape), obs_dtype)
    next_obs_mm = open_npy_memmap(next_obs_path, (N, *obs_shape), obs_dtype)
    act_mm = open_npy_memmap(act_path, (N, *act_shape), np.float32)
    rew_mm = open_npy_memmap(rew_path, (N,), np.float32)
    term_mm = open_npy_memmap(term_path, (N,), np.float32)

    print("Streaming replay buffer -> .npy files...")
    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        print(f"  [{start}:{end}]")

        obs_chunk = obs_src[start:end]
        next_obs_chunk = next_obs_src[start:end]

        if STORE_OBS_UINT8:
            # store exactly as uint8 to save disk (convert to float32 later per-batch)
            obs_mm[start:end] = obs_chunk.astype(np.uint8, copy=False)
            next_obs_mm[start:end] = next_obs_chunk.astype(np.uint8, copy=False)
        else:
            obs_mm[start:end] = obs_chunk.astype(np.float32)
            next_obs_mm[start:end] = next_obs_chunk.astype(np.float32)

        act_mm[start:end] = rb.actions[start:end].astype(np.float32)
        rew_mm[start:end] = rb.rewards[start:end].reshape(-1).astype(np.float32)
        term_mm[start:end] = rb.dones[start:end].reshape(-1).astype(np.float32)

    # Flush to disk
    del obs_mm, next_obs_mm, act_mm, rew_mm, term_mm

    print("\n✅ Done. Wrote:")
    for p in [obs_path, next_obs_path, act_path, rew_path, term_path]:
        print(" ", p)

    # Quick check (loads via mmap)
    obs = np.load(obs_path, mmap_mode="r")
    acts = np.load(act_path, mmap_mode="r")
    print("\nShapes check:")
    print("  observations:", obs.shape, obs.dtype)
    print("  actions:", acts.shape, acts.dtype)


if __name__ == "__main__":
    main()
