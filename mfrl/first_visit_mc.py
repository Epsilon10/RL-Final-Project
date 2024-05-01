import gymnasium as gym
import minigrid
import numpy as np
import argparse
import logging
from utils import env_extractor, make_required_dirs


def take_action(curr_state, returns, dims, eps):
    if curr_state not in returns or np.random.rand() < eps:
        return np.random.randint(dims)
    act = np.argmax(returns[curr_state])
    return act


def learn(env, test_env, dims, num_episodes, gamma, eps, extract_state):
    ep_returns = []
    ep_greedy_returns = []
    returns = {}
    visitation = {}

    for ep in range(num_episodes):
        env.reset()
        done = False

        states = []
        actions = []
        rewards = []
        while not done:
            curr_state = extract_state(env)
            act = take_action(curr_state, returns, dims, eps)
            _, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            states.append(curr_state)
            actions.append(act)
            rewards.append(rew)

        # Update all items
        ret = 0.
        step_num = len(states) - 1
        for state, act, rew in zip(reversed(states), reversed(actions), reversed(rewards)):
            ret *= gamma
            ret += rew
            if state in states[:step_num]:
                continue

            if state not in visitation:
                visitation[state] = np.zeros(dims)
                returns[state] = np.zeros(dims)
                visitation[state][act] = 1
                returns[state][act] = ret
            else:
                visits = visitation[state][act]
                visitation[state][act] = visits + 1
                returns[state][act] = (visits * returns[state][act] + ret) / (visits + 1)

            step_num -= 1

        ep_returns.append(ret)

        # Get greedy actions
        test_env.reset()
        done = False
        test_ret = 0.
        g = 1.
        while not done:
            _, rew, terminated, truncated, _ = test_env.step(take_action(extract_state(test_env), returns, dims, 0.))
            test_ret += rew * g
            g *= gamma
            done = terminated or truncated
        ep_greedy_returns.append(test_ret)
        
        logging.info(f"Ep {ep}:\tRet: {ret}\tTest: {test_ret}")

    return returns, ep_returns, ep_greedy_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-Unlock-v0")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=1.)
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

    returns, ep_returns, ep_greedy_returns = learn(env, test_env, env.action_space.n, args.num_episodes, args.gamma, args.eps, env_extractor(args.env))
    if args.save_returns is not None:
        np.save(args.save_returns, np.array(ep_returns))
    if args.save_greedy_returns is not None:
        np.save(args.save_greedy_returns, np.array(ep_greedy_returns))

