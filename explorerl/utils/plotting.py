import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def rolling_avg(rewards,sliding_window):
    avg = [np.mean(rewards[np.max(i-sliding_window,0):i+1]) for i in range(rewards.shape[0])]
    return np.array(avg)
    
def plot_metrics(stats_dict,sliding_window=10):
    episodes = np.array(stats_dict["episodes"])
    num_steps = np.array(stats_dict["num_steps"])
    rewards = np.array(stats_dict["rewards"])
    fig = plt.figure(1)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=.25)
    # episodic lengths
    plt.subplot(311)
    plt.plot(episodes,num_steps,color="blue")
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps')
    plt.title("Episodic Steps")
    # episodic rewards
    plt.subplot(312)
    plt.plot(episodes,rewards,color="red")
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title("Episodic Rewards")
    # episodic average rewards
    plt.subplot(313)
    plt.plot(episodes,rolling_avg(rewards,sliding_window),color="green")
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Rewards')
    plt.title("Episodic Smoothed Rewards - Window size ({})".format(str(sliding_window)))
    plt.show()
    
    