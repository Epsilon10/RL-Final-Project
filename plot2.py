import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 3000

def pad_1d_with_nan(arr, length):
    padded_arr = np.full(length, np.nan)
    padded_arr[:len(arr)] = arr
    return padded_arr

algos = ['1st_Visit_MC', 'Backwards_Q', 'Sarsa_5', 'MC', 'Sarsa_20', 'Sarsa', 'Q']

def gen_plots(env):
    timestamps = np.arange(3000)

   # plt.show()
   
    fig, ax = plt.subplots()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.5))

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    first_mc = np.load('final/RedBlueDoor/1st_Visit_MC.npy')[:3000]
    backwards_q = np.load('final/RedBlueDoor/Backwards_Q.npy')[:3000]
    mc = np.load('final/RedBlueDoor/MC.npy')[:3000]
    q = np.load('final/RedBlueDoor/Q.npy')[:3000]
    sarsa_5 = np.load('final/RedBlueDoor/Sarsa_5.npy')[:3000]
    sarsa_20 = np.load('final/RedBlueDoor/Sarsa_20.npy')[:3000]
    sarsa = np.load('final/RedBlueDoor/Sarsa.npy')[:3000]
    n_final = pad_1d_with_nan(np.load('final/RedBlueDoor/n_final.npy'), 3000)
    p_final = pad_1d_with_nan(np.load('final/RedBlueDoor/p_final.npy'), 3000)


    print("FIRST MC: ", first_mc)
    # Plot the data
    ax.plot(timestamps, first_mc, label='First Visit MC', color='#FF8C00')
    ax.plot(timestamps, backwards_q, label='Backwards Q', color='#1E90FF')
    ax.plot(timestamps, mc, label='MC', color='#7b583e')
    ax.plot(timestamps, q, label='Q', color='#f5874b')
    ax.plot(timestamps, sarsa_5, label='Sarsa 5', color='#b12e83')
    ax.plot(timestamps, sarsa_20, label='Sarsa 20', color='#29915c')
    ax.plot(timestamps, sarsa, label='Sarsa', color='#734f8b')
    ax.plot(timestamps, n_final, label='Dyna', color='#f2c335')
    ax.plot(timestamps, p_final, label='Dyna PS', color='#2c78a6')

    # Add gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Add legend
    ax.legend(loc='upper right')


    # Set labels and title
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Return')
    ax.set_title('RedBlueDoor')

    plt.savefig(f'RedBlueDoor.png')
    # Save or show the plot
    #plt.show()

if __name__ == "__main__":
    gen_plots(None)