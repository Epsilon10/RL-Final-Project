import numpy as np
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue
from math import sqrt

KEY = 5
ALPHA = 0.05
GAMMA = .98
THETA = 0
KAPPA = 0

def is_key(o, pos):
    return int(o['image'][pos][0] == KEY)

def random_argmax(arr):
    ans =  np.random.choice(np.flatnonzero(arr == arr.max()))
    return ans

def e_greedy_policy(s, q, epsilon=0.3):
    if random.random() < epsilon:
        return random.randint(0, len(q[s]) - 1)
    else:
        return random_argmax(q[s])

def state_from_obs(o, agent_pos):
    return agent_pos + (o['direction'], is_key(o, agent_pos))

def dyna_ps(env, name, num_planning_steps, num_episodes, state_shape, nA, lr,epsilon_init,gamma, extract_state_func):
    # states = []
    rew = []
    state_dims = len(state_shape)
    
    num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height
    sa_state = state_shape + (nA, )

    q = np.zeros(sa_state)

    model = np.full(sa_state + (1 + state_dims, ), -3, dtype=np.float32)

    pq = PriorityQueue()

    pred = {}
    epsilon = epsilon_init
    cum_rew = 0
    avg_rew = []
    rolling_avg = []
    rolling_avg_sum = 0

    last_taken = np.zeros(sa_state)
    ts = 0

    for i in range(num_episodes):
        print("EP NUMBER: ", i)
        print("PQ SIZE: ", pq.qsize())
        o, _ = env.reset()
        new_s = extract_state_func(env, o['direction'])
        epsilon = max(epsilon - .001,.01)
        print("EPSILON: ", epsilon)

       # last_tried = np.zeros((num_cells, 4, num_cells, 2, num_cells, 2, nA))
        while True:
            s = new_s
            a = e_greedy_policy(s, q, epsilon)

           # last_tried[s][a] = ts

            o, r, terminated, truncated, _ = env.step(a)
            new_s = extract_state_func(env, o['direction'])

            tmp_diff = abs(r + gamma * q[new_s].max() - q[s][a])
            model[s][a][0] = r
            model[s][a][1:] = list(new_s)
            # print("TMP DIFF: ", tmp_diff)

            last_taken[s][a] = ts

            if tmp_diff > THETA:
                pq.put((-tmp_diff, (s,a)))

            # states.append(s)
            if new_s not in pred.keys():
                pred[new_s] = [(s, a)]
            else:
                if (s,a) not in pred[new_s]:
                    pred[new_s].append((s,a))

            for _ in range(num_planning_steps):
                if pq.empty():
                    break
                res = pq.get()

                top_s, top_a = res[1]
                
                model_out = model[top_s][top_a]
                model_r = model_out[0]

                model_s_prime = tuple(np.array(model_out[1:], dtype=np.int32))

                delta = (model_r + gamma * q[model_s_prime].max() - q[top_s][top_a])

                q[top_s][top_a] = q[top_s][top_a] + lr * delta

                if top_s not in pred.keys():
                    continue

                for _, (s_pred, a_pred) in enumerate(pred[top_s]):
                    r_pred = model[s_pred][a_pred][0]
                   # print("LAST TAKEN: ", last_taken[s_pred][a_pred])
                    tmp_diff_pred = abs(r_pred + gamma * q[top_s].max() - q[s_pred][a_pred] + KAPPA * sqrt(ts - last_taken[s_pred][a_pred])) 
                    if tmp_diff_pred > THETA:
                        pq.put((-tmp_diff_pred, (s_pred, a_pred)))
               
            if terminated or truncated:
                rew.append(r)
                cum_rew += r
                avg_rew.append(cum_rew / num_episodes)
                if i < 10:
                    rolling_avg.append(0)
                    rolling_avg_sum += r
                else:
                    ans = sum(rew[len(rew) - 10:]) / 10
                    print("ANS: ", ans)
                    rolling_avg.append(sum(rew[len(rew) - 10:]) / 10)

                print("REWARD: ", r)
                print("STATE: ", s)
                print("Q: ", q[s][a])
                break

            ts += 1
            
           # states.append(s)
    return rolling_avg
    """
    timestamps = np.arange(len(rew))
    plt.scatter(timestamps, rolling_avg)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.title(f"Prioritized Sweeping")
    plt.grid(True)
    plt.savefig(f'name={name}_ps_n={num_planning_steps}_lr={lr}_epsilon={epsilon_init}.png')
    """


