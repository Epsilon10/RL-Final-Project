import subprocess
import time
import argparse


commands = []

for seed in range(3):
    for env in ["MiniGrid-Unlock-v0", "MiniGrid-RedBlueDoors-8x8-v0", "MiniGrid-Dynamic-Obstacles-5x5-v0", "MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-UnlockPickup-v0"]:
        eps = 0.6 if "Dynamic" not in env else 0.2
        alpha = 1e-2
        path = f"{env}/q/eps_{eps}_alpha_{alpha}/seed_{seed}"
        commands.append(
            ["python", "q.py",
                "--env", env,
                "--num_episodes", "10000",
                "--seed", str(seed),
                "--eps", str(eps),
                "--alpha", str(alpha),
                "--gamma", "0.999",
                "--log_to", f"{path}/log.log",
                "--save_returns", f"{path}/returns.npy",
                "--save_greedy_returns", f"{path}/greedy_returns.npy"
            ]
        )

        eps = 0.6 if "Dynamic" not in env else 0.2
        alpha = 5e-2
        path = f"{env}/backwards_q/eps_{eps}_alpha_{alpha}/seed_{seed}"
        commands.append(
            ["python", "backwards_q.py",
                "--env", env,
                "--num_episodes", "10000",
                "--seed", str(seed),
                "--eps", str(eps),
                "--alpha", str(alpha),
                "--gamma", "0.999",
                "--log_to", f"{path}/log.log",
                "--save_returns", f"{path}/returns.npy",
                "--save_greedy_returns", f"{path}/greedy_returns.npy"
            ]
        )

        eps = 0.6 if "Dynamic" not in env else 0.2
        path = f"{env}/mc/eps_{eps}/seed_{seed}"
        commands.append(
            ["python", "mc.py",
                "--env", env,
                "--num_episodes", "10000",
                "--seed", str(seed),
                "--eps", str(eps),
                "--gamma", "0.999",
                "--log_to", f"{path}/log.log",
                "--save_returns", f"{path}/returns.npy",
                "--save_greedy_returns", f"{path}/greedy_returns.npy"
            ]
        )

        eps = 0.6 if "Dynamic" not in env else 0.2
        path = f"{env}/first_visit_mc/eps_{eps}/seed_{seed}"
        commands.append(
            ["python", "first_visit_mc.py",
                "--env", env,
                "--num_episodes", "10000",
                "--seed", str(seed),
                "--eps", str(eps),
                "--gamma", "0.999",
                "--log_to", f"{path}/log.log",
                "--save_returns", f"{path}/returns.npy",
                "--save_greedy_returns", f"{path}/greedy_returns.npy"
            ]
        )

        eps = 0.6 if "Dynamic" not in env else 0.2
        alpha = 1e-2
        path = f"{env}/sarsa/eps_{eps}_alpha_{alpha}/seed_{seed}"
        commands.append(
            ["python", "sarsa.py",
                "--env", env,
                "--num_episodes", "10000",
                "--seed", str(seed),
                "--eps", str(eps),
                "--alpha", str(alpha),
                "--gamma", "0.999",
                "--log_to", f"{path}/log.log",
                "--save_returns", f"{path}/returns.npy",
                "--save_greedy_returns", f"{path}/greedy_returns.npy"
            ]
        )

        for n in [5, 20]:
            eps = 0.6 if "Dynamic" not in env else 0.2
            alpha = 1e-2
            path = f"{env}/sarsa_{n}/eps_{eps}_alpha_{alpha}/seed_{seed}"
            commands.append(
                ["python", "sarsa_n.py",
                    "--env", env,
                    "--num_episodes", "10000",
                    "--n", str(n),
                    "--seed", str(seed),
                    "--eps", str(eps),
                    "--alpha", str(alpha),
                    "--gamma", "0.999",
                    "--log_to", f"{path}/log.log",
                    "--save_returns", f"{path}/returns.npy",
                    "--save_greedy_returns", f"{path}/greedy_returns.npy"
                ]
            )


import os
# Filter out anything that's already run
commands = [cmd for cmd in commands if not os.path.isfile(cmd[-1])]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel_procs", type=int, default=4)

    args = parser.parse_args()

    processes = []

    try:
        while True:
            terminated_processes = [(str_cmd, proc) for str_cmd, proc in processes if proc.poll() is not None]
            processes = [(str_cmd, proc) for str_cmd, proc in processes if proc.poll() is None]

            for str_cmd, _ in terminated_processes:
                print(f"FINISHED: {str_cmd}")

            while len(processes) < args.parallel_procs and len(commands) != 0:
                str_cmd = " ".join(commands[0])
                proc = subprocess.Popen(commands[0], stdout=subprocess.DEVNULL)
                print(f"STARTED: {str_cmd}")
                processes.append((str_cmd, proc))
                commands = commands[1:]

            if len(processes) == 0 and len(commands) == 0:
                break

            time.sleep(10)

    except KeyboardInterrupt:
        print("\n")
        for str_cmd, proc in processes:
            print(f"KILLING: {str_cmd}")
            proc.kill()


