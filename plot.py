import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 1000

def gen_plots(data_1, data_2):
    timestamps = np.arange(len(data_1))
   # plt.show()
   
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot the data
    ax.plot(timestamps, data_1, label='Normal', color='#FF8C00')
    ax.plot(timestamps, data_2, label='Prioritized Sweeping', color='#1E90FF')


    # Add gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Add legend
    ax.legend()

    # Set labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Return')
    ax.set_title('Dyna on Red Blue Door')

    plt.savefig(f'lookatme.png')
    # Save or show the plot
    #plt.show()

if __name__ == "__main__":
    n_final = np.zeros(NUM_EPISODES, dtype=np.float32)
    p_final = np.zeros(NUM_EPISODES, dtype=np.float32)

    for i in range(1):
        n = np.load(f"normal_{i}.npy")
        p = np.load(f"ps_{i}.npy")

        n_final += n
        p_final += p
    
    n_final /= 1.0
    p_final /= 1.0

    np.save('n_final.npy', n_final)
    np.save('p_final.npy', p_final)

    gen_plots(n_final, p_final)