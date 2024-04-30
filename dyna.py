import numpy as np
import random
import matplotlib.pyplot as plt

# STATE SPACE IS (7,7,4, 2) <=> (x,y,dir, key present)
STATE_SPACE_SHAPE = [7,7,4,2]
KEY = 5

# GAMMA = .98
GAMMA = .98

def is_key(o, pos):
    return int(o['image'][pos][0] == KEY)

def random_argmax(arr):
    ans =  np.random.choice(np.flatnonzero(arr == arr.max()))
    return ans

#changing epsilon from .01 to .2
def e_greedy_policy(s, q, epsilon=.01):
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

def dyna_normal(env, name, num_planning_steps, num_episodes, state_shape, nA, lr,epsilon_init, gamma, extract_state_func):
    states = []
    rew = []
    state_dims = len(state_shape)
    
    num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height
    sa_state = state_shape + (nA, )

    q = np.zeros(sa_state)

    model = np.full(sa_state + (1 + state_dims, ), -3, dtype=np.float32)
    rolling_avg = []
    rolling_avg_sum = 0

    for i in range(num_episodes):
        print("EP NUMBER: ", i)
        o, _ = env.reset()
        new_s = extract_state_func(env, o['direction'])

        while True:
            s = new_s
            a = e_greedy_policy(s, q)
        # print("A: ", a)

            o, r, terminated, truncated, _ = env.step(a)
            new_s = extract_state_func(env, o['direction'])
        # print("NEW S: ", new_s)
            q[s][a] = q[s][a] + lr * (r + gamma * q[new_s].max() - q[s][a])
            model[s][a][0] = r
            model[s][a][1:] = list(new_s)

           # print(f"S: {s} and A: {a}")

            if r != 0:
                print(f"AMONGUS: {r}, then {model[s][a]}")

            states.append(s)

            for _ in range(num_planning_steps):
                rand_state_idx = random.randint(0, len(states) - 1)
                rand_state = states[rand_state_idx]

                possible_actions_for_state = []
                for ac, sr_pair in enumerate(model[rand_state]):
                    if sr_pair[0] != -3:
                        possible_actions_for_state.append(ac)
                
                # print("POSSIBLE ACTIONS FOR STATE: ", possible_actions_for_state)

                rand_action = np.random.choice(possible_actions_for_state)

                model_out = model[rand_state][rand_action]
                model_r = model_out[0]
               # if model_r != 0:
                    #print("WTFFFF")
                model_s_prime = tuple(np.array(model_out[1:], dtype=np.int32))
               
                #print("Q SHAPE: ", q.shape)
              #  print("MODEL R: ", model_r)
               # print("Q MAX: ", q[model_s_prime].max())
                delta = (model_r + gamma * q[model_s_prime].max() - q[rand_state][rand_action])
              #  print("DELTA: ", delta)
                #print("Q BEFORE: ", q[rand_state][rand_action])
                q[rand_state][rand_action] = q[rand_state][rand_action] + lr * delta
               
               # print(f"AFTER Q UPDATE FOR {rand_state} and {rand_action} is {q[rand_state][rand_action]}")

            if terminated or truncated:
                rew.append(r)
                # cum_rew += r
                # avg_rew.append(cum_rew / num_episodes)
                if i < 10:
                    rolling_avg.append(0)
                    rolling_avg_sum += r
                else:
                    rolling_avg_sum -= rolling_avg[i - 10]
                    rolling_avg_sum += r
                    ans = sum(rew[len(rew) - 10:]) / 10
                    print("ANS: ", ans)
                    rolling_avg.append(sum(rew[len(rew) - 10:]) / 10)

                print("REWARD: ", r)
                print("STATE: ", s)
                print("Q: ", q[s][a])
                break
                
           # states.append(s)
    return rolling_avg
    """
    timestamps = np.arange(len(rew))
    plt.scatter(timestamps, rolling_avg)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.title(f"Dyna Reward Planning Steps N={num_planning_steps}")
    plt.grid(True)
    plt.show()
    
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot the data
    ax.plot(timestamps, rolling_avg, label='Dyna', color='#FF8C00')

    # Add gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Add legend
    ax.legend()

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('NORMAL DYNA')

    # Save or show the plot
    plt.show()
    """

