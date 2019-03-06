import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display



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

def play_render(env,agent,steps=1000):
    observation = env.reset()
    frames = []
    policy = agent.greedy()
    for t in range(steps):
        frames.append(env.render(mode = 'rgb_array'))
        action, _ = policy(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    display_frames_as_gif(frames)