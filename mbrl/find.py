import glob
import numpy as np

directory = 'mfrl'
file_name = 'returns.npy'
files = glob.glob(f"{directory}/**/{file_name}", recursive=True)

envs = ['UnlockEnv', 'RedBlueDoor', 'Dynamic_Obstacle']
algos = ['1st_Visit_MC', 'Backwards_Q', 'Sarsa_5', 'MC', 'Sarsa_20', 'Sarsa', 'Q']
width = 3 * len(algos)

def rolling_avg(data):
    r = []
    print("DATA: ", data)
    for i in range(len(data)):
        if i < 10:
            r.append(0)
        else:
            r.append(sum(data[i - 10:i]) / 10.0)
    return np.array(r)

def proc_env(env_no):
    for i in range(len(algos)):
        ret0 = rolling_avg(np.load(files[width*env_no + i*3]))
        ret1 = rolling_avg(np.load(files[width*env_no + i*3 + 1]))
        ret2 = rolling_avg(np.load(files[width*env_no + i*3 + 2]))
        print("RET O: ", ret0[3000])

        algo = algos[i]
        #name = f"{envs[env_no]}_{algo}.npy"
        #print("NAME: ", name)
        #print(f"FILES: {files[width*env_no + i*3]}, {files[width*env_no + i*3 + 1]}, {files[width*env_no + i*3 + 2]}")

        final = (ret0 + ret1 + ret2) / 3.0
        print("FINAL: ", final)
        np.save(f"final/{envs[env_no]}/{algo}.npy", final)

if __name__ == "__main__":
    for i in range(len(envs)):
        proc_env(i)
