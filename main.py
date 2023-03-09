import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DRL_Functions import * 


problem = "Pendulum-v1"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

trial_params = []

tau = .4
        
for i in range(2):
    hyperparameters = {
        'env': "Pendulum-v1",
        'std_dev': 0.2,
        'actor_lr': 0.001,
        'critic_lr': 0.002,
        'gamma': .99,
        'tau': tau,
        'total_episodes': 100
    }
    trial_params.append(hyperparameters)

    model = ActorCritic(hyperparameters['env'], 
                hyperparameters['std_dev'], 
                hyperparameters['actor_lr'],
                hyperparameters['critic_lr'],
                hyperparameters['gamma'],
                hyperparameters['tau'],
                hyperparameters['total_episodes']
                )
    model.train_actor_critic()
    model.display_episode(i)
    model.write_hyperparameters_to_file()
    model.save_weights()

    tau -= .04

        
print(trial_params)
