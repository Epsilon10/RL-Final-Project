import numpy as np
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue

# STATE SPACE IS (7,7,4, 2) <=> (x,y,dir, key present)
STATE_SPACE_SHAPE = [7,7,4,2]
KEY = 5
ALPHA = 0.05
GAMMA = 1
N = 10
THETA = 0

STATE_DIMS = 5
NUM_EPISODES = 500

def is_key(o, pos):
    return int(o['image'][pos][0] == KEY)

def random_argmax(arr):
    ans =  np.random.choice(np.flatnonzero(arr == arr.max()))
    return ans

#changing epsilon from .01 to .2
def e_greedy_policy(s, q, epsilon=.5):
    if random.random() < epsilon:
        return random.randint(0, len(q[s]) - 1)
    else:
        return random_argmax(q[s])

def state_from_obs(o, agent_pos):
    return agent_pos + (o['direction'], is_key(o, agent_pos))

def extract_state(env, dir):
    width = env.unwrapped.grid.width
    height = env.unwrapped.grid.height
    grid = env.unwrapped.grid.grid
    grid_items = [item.type if item is not None else "" for item in grid]
    agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
    key = grid_items.index("key") if "key" in grid_items else width * height
    door = grid_items.index("door") if "door" in grid_items else width * height
    open_door = int(grid[door].is_open)
    # print("AGENT POS: ", env.unwrapped.agent_pos)
    return (agent, dir, key, door, open_door)

def dyna(env):
    states = []
    rew = []

    nA = 7
    
    num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height

    q = np.zeros((num_cells, 4, num_cells + 1, num_cells + 1, 2, nA))

    model = np.full((num_cells, 4, num_cells + 1, num_cells + 1, 2, nA, 1 + STATE_DIMS), -1, dtype=np.float32)
    pq = PriorityQueue()
    pred = {}

    for i in range(NUM_EPISODES):
        print("EP NUMBER: ", i)
        o, _ = env.reset()
        new_s = extract_state(env, o['direction'])

        while True:
            s = new_s
            a = e_greedy_policy(s, q)
        # print("A: ", a)

            o, r, terminated, truncated, _ = env.step(a)
            new_s = extract_state(env, o['direction'])
        # print("NEW S: ", new_s)
            # q[s][a] = q[s][a] + ALPHA * (r + GAMMA * q[new_s].max() - q[s][a])
            model[s][a][0] = r
            model[s][a][1:] = list(new_s)

            if new_s not in pred.keys():
                pred[new_s] = [(s,a)]
            else:
                if (s,a) not in pred[new_s]:
                    pred[new_s].append((s,a))

            P = abs(r + GAMMA * q[new_s].max() - q[s][a])
            if P > THETA:
                print("ADDING: ",(s,a))
                pq.put((-P, (s, a)))

           # if r != 0:
               # print(f"AMONGUS: {r}, then {model[s][a]}")

            states.append(s)

            for _ in range(N):
                if pq.empty():
                    break
                #print("FIRST: ", pq[0])
                x = pq.get()
               # print("X: ", x)
                top_s = x[0][0]
                top_a = x[0][1]
                print(f"TOP S: {top_s} and TOP A: {top_a} with TDE {x[0]}")

                out = model[s][a]
                top_r = out[0]
                top_s_prime = tuple(np.array(out[1:], dtype=np.int32))
                #print("TOP S PRIME: ", top_s_prime)
                q[top_s][top_a] = q[top_s][top_a] + ALPHA * (top_r + GAMMA * q[top_s_prime].max() - q[top_s][top_a])

                if top_s not in pred.keys():
                    continue

                for pred_s, pred_a in pred[top_s]:
                    pred_r = model[pred_s][pred_a][0]
                    pred_P = abs(pred_r + GAMMA * q[top_s].max() - q[pred_s][pred_a])
                    if pred_P > THETA:
                        print("ADDING SUS: ", (pred_s, pred_a))
                        pq.put((-pred_P, (pred_s, pred_a)))
                
               # print(f"AFTER Q UPDATE FOR {rand_state} and {rand_action} is {q[rand_state][rand_action]}")

            if terminated or truncated:
                rew.append(r)
                print("REWARD: ", r)
                print("STATE: ", s)
                print("Q: ", q[s][a])
                break
                
           # states.append(s)
    timestamps = np.arange(len(rew))
    plt.scatter(timestamps, rew)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.title(f"Dyna Reward Planning Steps N={N}")
    plt.grid(True)
    plt.show()
    


