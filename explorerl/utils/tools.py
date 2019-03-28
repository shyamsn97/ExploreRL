import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def auto_import():
    import numpy as np
    import torch
    import tensorflow as tf
    import gym
    import sys
    from tqdm import tqdm

##### Animation #####
def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))

def play_render(env,agent,episodes=1,steps=1000,display=False,gif=True):
    policy = agent.test_policy()
    for ep in range(episodes):
        frames = []
        observation = env.reset()
        total_reward = 0
        for t in range(steps):
            observation = agent.featurize_state(observation)
            if display:
                env.render()
            if gif:
                frames.append(env.render(mode = 'rgb_array'))
            action, values = policy(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        print("Total reward for episode {}: {}".format(ep,total_reward))
        if gif:
            display_frames_as_gif(frames)
    env.close()

    