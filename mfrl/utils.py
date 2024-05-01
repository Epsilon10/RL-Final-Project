def make_required_dirs(files):
    import os
    for filename in files:
        if filename is not None:
            os.makedirs(os.path.dirname(filename), exist_ok=True)


def env_extractor(env_name):
    if env_name == "MiniGrid-Unlock-v0":
        return extract_state_unlock
    elif env_name == "MiniGrid-UnlockPickup-v0":
        return extract_state_unlockpickup
    elif env_name == "MiniGrid-RedBlueDoors-8x8-v0":
        return extract_state_redbluedoor
    elif env_name == "MiniGrid-Dynamic-Obstacles-8x8-v0" or env_name == "MiniGrid-Dynamic-Obstacles-5x5-v0":
        return extract_state_dynamicobstacles

    raise Exception("unsupported env")


def extract_state_unlock(env):
    width = env.unwrapped.grid.width
    height = env.unwrapped.grid.height
    grid = env.unwrapped.grid.grid
    grid_items = [item.type if item is not None else "" for item in grid]
    agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
    key = grid_items.index("key") if "key" in grid_items else width * height
    door = grid_items.index("door") if "door" in grid_items else width * height
    open_door = int(grid[door].is_open)
    direction = env.unwrapped.agent_dir
    return (agent, key, door, open_door, direction)


def extract_state_unlockpickup(env):
    width = env.unwrapped.grid.width
    height = env.unwrapped.grid.height
    grid = env.unwrapped.grid.grid
    grid_items = [item.type if item is not None else "" for item in grid]
    agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
    key = grid_items.index("key") if "key" in grid_items else width * height
    door = grid_items.index("door") if "door" in grid_items else width * height
    box = grid_items.index("box") if "box" in grid_items else width * height
    open_door = int(grid[door].is_open)
    direction = env.unwrapped.agent_dir
    return (agent, key, door, open_door, box, direction)


def extract_state_redbluedoor(env):
    width = env.unwrapped.grid.width
    height = env.unwrapped.grid.height
    grid = env.unwrapped.grid.grid
    grid_items = [item.type if item is not None else "" for item in grid]
    agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
    first_door = grid_items.index("door")
    second_door = first_door + 1 + grid_items[first_door + 1:].index("door")
    first_open_door = int(grid[first_door].is_open)
    second_open_door = int(grid[second_door].is_open)
    direction = env.unwrapped.agent_dir
    return (agent, first_door, second_door, first_open_door, second_open_door, direction)

def extract_state_dynamicobstacles(env):
    width = env.unwrapped.grid.width
    grid = env.unwrapped.grid.grid
    agent = env.unwrapped.agent_pos[1] * width + env.unwrapped.agent_pos[0]
    direction = env.unwrapped.agent_dir
    grid_items = [item.type if item is not None else "" for item in grid]

    state = [agent, direction]
    for i, s in enumerate(grid_items):
        if s == "ball":
            state.append(i)

    return tuple(state)
