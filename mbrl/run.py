#
#from ps_dyna_final import dyna
from dyna import dyna_normal
from ps_dyna_final import dyna_ps
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import argparse


# RED BLUE DOOR ENV STATE SHAPE = (num_cells, 4, num_cells, 2, 2, num_cells, 2,2)
# UNLOCK ENV STATe SHAPE = (num_cells, 4, num_cells + 1, num_cells + 1, 2)

def RED_BLUE_STATE_SHAPE(num_cells):
   return (num_cells, 4, num_cells, 2, 2, num_cells, 2,2)

def UNLOCK_ENV_STATE_SHAPE(num_cells):
   return (num_cells, 4, num_cells + 1, num_cells + 1, 2)

def DYN_ENV_STATE_SHAPE(num_cells):
   return (num_cells, 4, num_cells, num_cells, num_cells)

def UNLOCK_PICKUP_STATE_SHAPE(num_cells):
   return (num_cells, 4, num_cells + 1, num_cells + 1, 2, num_cells + 1)

def extract_state_red_blue_door(env, dir):
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

def extract_state_dyn_ob(env, dir):
   width = env.unwrapped.grid.width
   height = env.unwrapped.grid.height
   grid = env.unwrapped.grid.grid
   grid_items = [item.type if item is not None else "" for item in grid]
   agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
   goal = grid_items.index("goal") if "goal" in grid_items else width * height
   ball_info = []
   for i, item in enumerate(grid_items):
      if item == 'ball':
         ball_info.append(i)
   
   ball_0 = ball_info[0]
   ball_1 = ball_info[1]

   return (agent, dir, goal, ball_0, ball_1)
    
def extract_state_unlock(env, dir):
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

def extract_state_pickup(env, dir):
    width = env.unwrapped.grid.width
    height = env.unwrapped.grid.height
    grid = env.unwrapped.grid.grid
    grid_items = [item.type if item is not None else "" for item in grid]
    agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
    key = grid_items.index("key") if "key" in grid_items else width * height
    door = grid_items.index("door") if "door" in grid_items else width * height
    box = grid_items.index("box") if "box" in grid_items else width * height
    open_door = int(grid[door].is_open)
    # print("AGENT POS: ", env.unwrapped.agent_pos)
    return (agent, dir, key, door, open_door, box)

def unlock_env():
   env = gym.make("MiniGrid-Unlock-v0")
   num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height

   normal_avg = dyna_normal(env=env, name="normal_unlock_env", num_planning_steps=10, num_episodes=1000, state_shape=UNLOCK_ENV_STATE_SHAPE(num_cells), nA=7, lr=0.05,epsilon_init=.05,gamma=.98, extract_state_func=extract_state_unlock)
   ps_avg = dyna_ps(env=env, name="ps_unlock_env", num_planning_steps=25, num_episodes=1000, state_shape=UNLOCK_ENV_STATE_SHAPE(num_cells), nA=7, lr=0.05,epsilon_init=.5,gamma=.98, extract_state_func=extract_state_unlock)
   return normal_avg, ps_avg

def red_blue_env():
   env = gym.make("MiniGrid-RedBlueDoors-6x6-v0")
   num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height

   normal_avg = dyna_normal(env=env, name="normal_red_blue_env", num_planning_steps=20, num_episodes=1000, state_shape=RED_BLUE_STATE_SHAPE(num_cells), nA=7, lr=0.05,epsilon_init=.5, gamma=.98, extract_state_func=extract_state_red_blue_door)
   ps_avg = dyna_ps(env=env, name="ps_red_blue_env", num_planning_steps=20, num_episodes=1000, state_shape=RED_BLUE_STATE_SHAPE(num_cells), nA=7, lr=0.15,epsilon_init=.5, gamma=.98, extract_state_func=extract_state_red_blue_door)
   return normal_avg, ps_avg

def dyn_ob_env():
   env = gym.make("MiniGrid-Dynamic-Obstacles-5x5-v0")
   num_cells = env.unwrapped.grid.width * env.unwrapped.grid.height

   normal_avg = dyna_normal(env=env, name="normal_dyn_env", num_planning_steps=10, num_episodes=2000, state_shape=DYN_ENV_STATE_SHAPE(num_cells), nA=3, lr=0.05,epsilon_init=.3, gamma=.98, extract_state_func=extract_state_dyn_ob)
   ps_avg = dyna_ps(env=env, name="ps_dyn_env", num_planning_steps=30, num_episodes=2000, state_shape=DYN_ENV_STATE_SHAPE(num_cells), nA=3, lr=0.01,epsilon_init=.3, gamma=.98, extract_state_func=extract_state_dyn_ob)
   return normal_avg, ps_avg

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Process some integers.')

# Add the argument
   parser.add_argument('mode', choices=['unlock', 'redbluedoor', 'dynamic'], help='Choose a mode')

   # Parse the arguments
   args = parser.parse_args()

   name = args.mode
   if name == 'unlock':
      normal_avg, ps_avg = unlock_env()
   elif name== 'redbluedoor':
      normal_avg, ps_avg = red_blue_env()
   elif name == 'dynamic':
      normal_avg, ps_avg = dyn_ob_env()

   timestamps = np.arange(len(normal_avg))
   plt.show()
   
   fig, ax = plt.subplots()
   fig.patch.set_facecolor('white')
   ax.set_facecolor('white')

   # Plot the data
   ax.plot(timestamps, normal_avg, label='Normal', color='#FF8C00')
   ax.plot(timestamps, ps_avg, label='Prioritized Sweeping', color='#1E90FF')

   # Add gridlines
   ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

   # Add legend
   ax.legend()

   # Set labels and title
   ax.set_xlabel('Episode')
   ax.set_ylabel('Average Return')
   ax.set_title('Dyna')

   plt.savefig(f'Dyna.png')
   # Save or show the plot
   plt.show()