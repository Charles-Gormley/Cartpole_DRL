import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def create_state(problem : str):
    env = gym.make(problem)

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
        
def display_episode(avg_reward_list : list):
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()

def save_weights(iteration : int):
    # Save the weights
    actor_model.save_weights("pendulum_actor.h5")
    critic_model.save_weights("pendulum_critic.h5")

    target_actor.save_weights("pendulum_target_actor.h5")
    target_critic.save_weights("pendulum_target_critic.h5")