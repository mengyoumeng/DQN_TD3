def eval_response(td3_path: str, X: np.ndarray, y: np.ndarray, attack_embed: np.ndarray, n_samples: int = 2000):
    model = TD3.load(td3_path)
    env = ResponseEnv(X, y, attack_embed)

    rewards = []
    actions = []
    n = min(n_samples, len(X))

    for i in range(n):
        env._idx = i
        obs = env._obs(i)
        a, _ = model.predict(obs, deterministic=True)
        actions.append(np.array(a, dtype=np.float32).reshape(-1))

        out = env.step(a)
        if isinstance(out, tuple) and len(out) == 5:
            _, r, _, _, _ = out
        else:
            _, r, _, _ = out

        rewards.append(float(r))

    rewards = np.asarray(rewards, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    nan_cnt = int(np.isnan(rewards).sum())
    avg = float(np.nanmean(rewards))

    return {
        "avg_reward": avg,
        "nan_rewards": nan_cnt,
        "n_eval": int(n),
        "action_mean": actions.mean(axis=0).tolist() if len(actions) > 0 else None,
        "action_std": actions.std(axis=0).tolist() if len(actions) > 0 else None,
    }
