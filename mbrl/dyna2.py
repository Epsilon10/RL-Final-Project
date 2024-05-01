import numpy as np
import random
import matplotlib.pyplot as plt

# STATE SPACE IS (7,7,4, 2) <=> (x,y,dir, key present)
KEY = 5
GAMMA = 1
N = 5
ALPHA = 0.01

STATE_DIMS = 6
NUM_EPISODES = 4000

def random_argmax(arr):
    ans =  np.random.choice(np.flatnonzero(arr == arr.max()))
    return ans

def e_greedy_policy(s, q, epsilon=.5):
    if random.random() < epsilon:
        return random.randint(0, len(q[s]) - 1)
    else:
        return random_argmax(q[s])

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
    return (agent, dir, door_0[0], int(door_0[1] == 'red'), door_1[0], int(door_1[1] == 'red'))
    
def dyna(env):
    states = []
    rew = []

    nA = 7
    
    num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height

    q = np.zeros((num_cells, 4, num_cells, 2, num_cells, 2, nA))

    model = np.full((num_cells, 4, num_cells, 2, num_cells, 2, nA, 1 + STATE_DIMS), -1, dtype=np.float32)

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
            q[s][a] = q[s][a] + ALPHA * (r + GAMMA * q[new_s].max() - q[s][a])
            model[s][a][0] = r
            model[s][a][1:] = list(new_s)

            if r != 0:
                print(f"AMONGUS: {r}, then {model[s][a]}")

            states.append(s)

            for _ in range(N):
                rand_state_idx = random.randint(0, len(states) - 1)
                rand_state = states[rand_state_idx]

                possible_actions_for_state = []
                for i, sr_pair in enumerate(model[rand_state]):
                    if sr_pair[0] != -1:
                        possible_actions_for_state.append(i)
                
                #print("POSSIBLE ACTIONS FOR STATE: ", possible_actions_for_state)

                rand_action = np.random.choice(possible_actions_for_state)

                model_out = model[rand_state][rand_action]
                model_r = model_out[0]
               # if model_r != 0:
                    #print("WTFFFF")
                model_s_prime = tuple(np.array(model_out[1:], dtype=np.int32))
               
                delta = (model_r + GAMMA * q[model_s_prime].max() - q[rand_state][rand_action])
                # print("DELTA: ", delta)
                #print("Q BEFORE: ", q[rand_state][rand_action])
                #print("DELTA: ", delta)
                #print("REWARD: ", model_r)
                #print("Q MAX: ", q[model_s_prime].max())
                q[rand_state][rand_action] = q[rand_state][rand_action] + ALPHA * delta
               
                #print(f"AFTER Q UPDATE FOR {rand_state} and {rand_action} is {q[rand_state][rand_action]}")

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
    


