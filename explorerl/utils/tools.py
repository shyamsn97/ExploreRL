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
    fig = plt.figure("Animation",figsize=(7,5))
    ax = fig.add_subplot(111)

    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    ims = []
    npad = ((4, 4), (4, 4),(0,0))
    for i in range(len(frames)):
        img = np.pad(frames[i][0], pad_width=npad, mode='constant', constant_values=255.) #pad white 
        frame =  ax.imshow(img)   
        ax.axis('off')
        t = ax.annotate(frames[i][1],(1,1)) # add text
        t.set_fontsize(12)

        ims.append([frame,t]) # add both the image and the text to the list of artists 

    anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
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
                frames.append((env.render(mode = 'rgb_array'),"Cumulative reward for episode {} at time step {}: {}".format(ep,t,total_reward)))
            action, values = policy(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        print("Total reward for episode {}: {}".format(ep,total_reward))
        if gif:
            display_frames_as_gif(frames)
    env.close()

    