import gymnasium as gym
import minigrid
import numpy as np
import argparse
import logging
from utils import env_extractor, make_required_dirs


def take_action(state, q_func, dims, eps):
    if state not in q_func or np.random.rand() < eps:
        return np.random.randint(dims)
    act = np.argmax(q_func[state])
    return act


def learn(env, test_env, dims, num_episodes, gamma, n, alpha, eps, extract_state):
    ep_returns = []
    ep_greedy_returns = []
    q_func = {}

    for ep in range(num_episodes):
        env.reset()
        done = False

        t = 0
        T = 10**9
        rewards = []
        states = []
        acts = []

        curr_state = extract_state(env)
        states.append(curr_state)
        acts.append(take_action(curr_state, q_func, dims, eps))

        while True:
            if t < T:
                _, rew, terminated, truncated, _ = env.step(acts[t])
                done = terminated or truncated
                rewards.append(rew)
                states.append(extract_state(env))
                if done:
                    T = t + 1
                else:
                    acts.append(take_action(states[t + 1], q_func, dims, eps))

            tau = t + 1 - n

            if tau >= 0:
                G = sum([gamma ** i * r for i, r in enumerate(rewards[tau:min(tau + n, T)])])
                if tau + n < T:
                    last_q = q_func[states[tau + n]][acts[tau + n]] if states[tau + n] in q_func else 0.
                    G += gamma ** n * last_q

                if states[tau] not in q_func:
                    q_func[states[tau]] = np.zeros(dims)

                q_func[states[tau]][acts[tau]] += alpha * (G - q_func[states[tau]][acts[tau]])
                
            t += 1

            if tau == T - 1:
                break

        ret = sum([gamma ** i * rew for i, rew in enumerate(rewards)])
        ep_returns.append(ret)

        # Get greedy actions
        test_env.reset()
        done = False
        test_ret = 0.
        g = 1.
        while not done:
            _, rew, terminated, truncated, _ = test_env.step(take_action(extract_state(test_env), q_func, dims, 0.))
            test_ret += rew * g
            g *= gamma
            done = terminated or truncated
        ep_greedy_returns.append(test_ret)

        logging.info(f"Ep {ep}:\tRet: {ret}\tTest: {test_ret}")

    return q_func, ep_returns, ep_greedy_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-Unlock-v0")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_to", type=str)
    parser.add_argument("--save_returns", type=str)
    parser.add_argument("--save_greedy_returns", type=str)

    args = parser.parse_args()

    make_required_dirs([args.log_to, args.save_returns, args.save_greedy_returns])

    if args.log_to is not None:
        logging.basicConfig(filename=args.log_to, level=logging.INFO)

    np.random.seed(args.seed)

    env = gym.make(args.env)
    test_env = gym.make(args.env)

    q_func, ep_returns, ep_greedy_returns = learn(env, test_env, env.action_space.n, args.num_episodes, args.gamma, args.n, args.alpha, args.eps, env_extractor(args.env))
    if args.save_returns is not None:
        np.save(args.save_returns, np.array(ep_returns))
    if args.save_greedy_returns is not None:
        np.save(args.save_greedy_returns, np.array(ep_greedy_returns))

