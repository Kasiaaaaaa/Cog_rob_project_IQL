import numpy as np
from stable_baselines3 import SAC

def export_npz(model_path, rb_path, out_path):
    model = SAC.load(model_path, env=None)
    model.load_replay_buffer(rb_path)

    rb = model.replay_buffer

    # --- Extract camera observations ---
    obs = rb.observations["cam_image"]
    next_obs = rb.next_observations["cam_image"]

    # obs may come as (N, 1, 1, 100, 100) or similar
    # We want (N, 100, 100, 1)

    # Remove extra singleton dimensions
    obs = np.squeeze(obs)
    next_obs = np.squeeze(next_obs)

    # Ensure channel-last format
    if obs.ndim == 3:  # (N,100,100)
        obs = obs[..., None]
        next_obs = next_obs[..., None]

    # --- Actions ---
    actions = rb.actions
    actions = np.squeeze(actions)  # (N,3)

    # --- Rewards & terminals ---
    rewards = rb.rewards.reshape(-1).astype(np.float32)
    dones = rb.dones.reshape(-1).astype(np.float32)

    # --- Final sanity check ---
    assert obs.ndim == 4 and obs.shape[-1] == 1, f"Bad obs shape: {obs.shape}"
    assert actions.ndim == 2 and actions.shape[1] == 3, f"Bad action shape: {actions.shape}"

    np.savez(
        out_path,
        observations=obs.astype(np.uint8),
        actions=actions.astype(np.float32),
        rewards=rewards,
        next_observations=next_obs.astype(np.uint8),
        terminals=dones,
    )

    print("Saved:", out_path)
    print("observations:", obs.shape, obs.dtype)
    print("actions:", actions.shape, actions.dtype)
    print("rewards:", rewards.shape)
    print("next_observations:", next_obs.shape, next_obs.dtype)
    print("terminals:", dones.shape)


if __name__ == "__main__":
    export_npz(
        model_path="./checkpoints/sac_model_160000_steps.zip",
        rb_path="./checkpoints/final_replay_buffer_small.pkl",
        out_path="./checkpoints/iql_dataset.npz",
    )
