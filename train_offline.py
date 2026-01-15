import os
from typing import Tuple, Any

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import Dataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner
from rlenv import PegInHoleGymEnv


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_string('shape', 'circle', 'Peg/hole shape.')
flags.DEFINE_string('reward', 'old', 'Which rewared will be used.')
flags.DEFINE_string('dataset_path', 'peg_iql_dataset.npz', 'Path to offline dataset.')


config_flags.DEFINE_config_file(
    'config',
    'configs/mujoco_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, Any]:
    env = PegInHoleGymEnv(shape_type=FLAGS.shape, reward_typ=FLAGS.reward) # gui=False

    env = wrappers.EpisodeMonitor(env)
    #env = wrappers.SinglePrecision(env)

    # Modern seeding (Gymnasium / newer Gym)
    try:
        env.reset(seed=seed)
    except TypeError:
        # Older Gym reset() signature
        env.reset()

    # Also seed the action/observation spaces if available
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    # -------- dataset loading (memmap .npy) --------
    # FLAGS.dataset_path should be the directory containing the .npy files
    # e.g. --dataset_path ./iql_dataset_npy

    ds_dir = FLAGS.dataset_path

    observations = np.load(os.path.join(ds_dir, "observations.npy"), mmap_mode="r")
    actions = np.load(os.path.join(ds_dir, "actions.npy"), mmap_mode="r")
    rewards = np.load(os.path.join(ds_dir, "rewards.npy"), mmap_mode="r")
    next_observations = np.load(os.path.join(ds_dir, "next_observations.npy"), mmap_mode="r")
    terminals = np.load(os.path.join(ds_dir, "terminals.npy"), mmap_mode="r")

    # terminals is float32 {0,1} already in the converter
    masks = 1.0 - terminals

    # IMPORTANT: do NOT astype() the full arrays here (that copies into RAM)
    dataset = Dataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        masks=masks,
        dones_float=terminals,
        next_observations=next_observations,
        size=len(observations),
    )

    print("Dataset obs dtype/shape:", dataset.observations.dtype, dataset.observations.shape)
    print("Dataset actions min/max:", dataset.actions[:1000].min(), dataset.actions[:1000].max())


    # if 'antmaze' in FLAGS.env_name:
    #     dataset.rewards -= 1.0
    #     # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    #     # but I found no difference between (x - 0.5) * 4 and x - 1.0
    # elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
    #       or 'hopper' in FLAGS.env_name):
    #     normalize(dataset)

    return env, dataset


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                                str(FLAGS.seed)),
                                   write_to_disk=True)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    # Use dataset to initialize shapes (env obs is a Dict)
    init_batch = dataset.sample(1)
    init_obs = init_batch.observations          # (1, 100, 100, 1)
    init_act = init_batch.actions               # (1, 3)

    agent = Learner(FLAGS.seed,
                    init_obs,
                    init_act,
                    max_steps=FLAGS.max_steps,
                    **kwargs)
    
    ###############################################
    ###############################################
    # --- sanity check right after dataset creation ---
    b = dataset.sample(4)

    print("Batch obs:", type(b.observations), b.observations.dtype, b.observations.shape)
    print("Batch act:", type(b.actions), b.actions.dtype, b.actions.shape)
    print("Batch rew:", type(b.rewards), b.rewards.dtype, b.rewards.shape)
    print("Batch next_obs:", type(b.next_observations), b.next_observations.dtype, b.next_observations.shape)
    print("Batch masks:", type(b.masks), b.masks.dtype, b.masks.shape)

    # hard asserts for your env
    assert b.observations.shape[-3:] == (100, 100, 1), f"Bad obs shape: {b.observations.shape}"
    assert b.actions.shape[-1] == 3, f"Bad action shape: {b.actions.shape}"

    # JAX conversion check (optional)
    import jax.numpy as jnp
    _ = jnp.asarray(b.observations)
    _ = jnp.asarray(b.actions)
    print("JAX conversion OK")

    # THE real test: one update step
    info = agent.update(b)
    print("Update OK, info keys:", list(info.keys()))
    for k, v in info.items():
        try:
            print(f"  {k}: {float(v):.6f}")
        except Exception:
            print(f"  {k}: {v}")

    # stop here so you don't start long training yet
    # import sys
    # sys.exit(0)
    ###############################################
    ###############################################

    eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
