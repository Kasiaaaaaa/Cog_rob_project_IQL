from rlenv import PegInHoleGymEnv

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


#obs space: Dict('cam_image': Box(0, 255, (100, 100, 1), uint8))
#act space: Box(-0.005, 0.005, (3,), float32)