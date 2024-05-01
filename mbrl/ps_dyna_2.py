import numpy as np
import random
import matplotlib.pyplot as plt
from queue import PriorityQueue
from math import sqrt

# STATE SPACE IS (7,7,4, 2) <=> (x,y,dir, key present)
STATE_SPACE_SHAPE = [7,7,4,2]
KEY = 5
ALPHA = 0.05
GAMMA = 1
N = 50
THETA = 0.001
KAPPA = 0

STATE_DIMS = 8
NUM_EPISODES = 1000

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

def extract_state(env, dir):
    width = env.unwrapped.grid.width
    height = env.unwrapped.grid.height
    grid = env.unwrapped.grid.grid
    grid_items = [item.type if item is not None else "" for item in grid]
    agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
    door_info = []
    for i, item in enumerate(grid_items):
        if item == 'door':
            door_info.append((i, grid[i].color))
    door_0 = door_info[0]
    door_1 = door_info[1]
    return (agent, dir, door_0[0], int(door_0[1] == 'red'), int(grid[door_0[0]].is_open), door_1[0], int(door_1[1] == 'red'), int(grid[door_1[0]].is_open))

def dyna(env):
    # states = []
    rew = []

    nA = 7
    
    num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height

    q = np.zeros((num_cells, 4, num_cells, 2, 2, num_cells, 2,2, nA))

    model = np.full((num_cells, 4, num_cells, 2, 2, num_cells, 2,2, nA, 1 + STATE_DIMS), -1, dtype=np.float32)

    pq = PriorityQueue()

    pred = {}
    epsilon = .6

    for i in range(NUM_EPISODES):
        print("EP NUMBER: ", i)
        print("PQ SIZE: ", pq.qsize())
        o, _ = env.reset()
        new_s = extract_state(env, o['direction'])
        epsilon = max(epsilon - .001,.01)
        print("EPSILON: ", epsilon)

       # last_tried = np.zeros((num_cells, 4, num_cells, 2, num_cells, 2, nA))
        ts = 0

        while True:
            s = new_s
            a = e_greedy_policy(s, q, epsilon)

           # last_tried[s][a] = ts

            o, r, terminated, truncated, _ = env.step(a)
            new_s = extract_state(env, o['direction'])

            tmp_diff = abs(r + GAMMA * q[new_s].max() - q[s][a])
            model[s][a][0] = r
            model[s][a][1:] = list(new_s)
            # print("TMP DIFF: ", tmp_diff)

            if tmp_diff > THETA:
                pq.put((-tmp_diff, (s,a)))

            # states.append(s)
            if new_s not in pred.keys():
                pred[new_s] = [(s, a)]
            else:
                if (s,a) not in pred[new_s]:
                    pred[new_s].append((s,a))

            for i in range(N):
                #print("I: ",i)
                #print("PQ SIZE: ", pq.qsize())
                if pq.empty():
                    break
                res = pq.get()
                # print("TOP TD: ", res[0])
                top_s, top_a = res[1]
                
                model_out = model[top_s][top_a]
                model_r = model_out[0]

                model_s_prime = tuple(np.array(model_out[1:], dtype=np.int32))

                delta = (model_r + GAMMA * q[model_s_prime].max() - q[top_s][top_a])
               # print("DELTA: ", delta)
                #print("MAX Q: ", q[model_s_prime].max())
                #print(f"STATE: {s} AND NEW STATE: {new_s}")

               # old_q = q[top_s][top_a]
                q[top_s][top_a] = q[top_s][top_a] + ALPHA * delta
              #  if q[top_s][top_a] > 1:
                    #print(f"Q UPDATE OLD: {old_q} now {q[top_s][top_a]}")

               # print("PRED R: ", model_r)

                if top_s not in pred.keys():
                    continue
              #  print("PRED LEN: ", len(pred[top_s]))
                for i, (s_pred, a_pred) in enumerate(pred[top_s], N):
                    if i == N - 1:
                        break
                    r_pred = model[s_pred][a_pred][0]

                    tmp_diff_pred = abs(r_pred + GAMMA * q[top_s].max() - q[s_pred][a_pred])
                    if tmp_diff_pred > THETA:
                      #  print("TRAINING HERE:")
                        pq.put((-tmp_diff_pred, (s_pred, a_pred)))
               
            if terminated or truncated:
                rew.append(r)
                print("REWARD: ", r)
                print("STATE: ", s)
                print("Q: ", q[s][a])
                break

            ts += 1
                
           # states.append(s)
    timestamps = np.arange(len(rew))
    plt.scatter(timestamps, rew)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.title(f"Dyna Reward Planning With PS Steps N={N}")
    plt.grid(True)
    plt.show()
    


