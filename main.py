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

tau_values = [.0001, .001,  .01, .1]
std_dev_values = [.02, .2, .4]
actor_critic_values = [[.0001, .0002], [.001, .002],  [.01, .02], [.1, .2]]

i = 0 
for tau in tau_values:
    for std_dev in std_dev_values:
        for actor_critic_value in actor_critic_values:
            actor_lr = actor_critic_value[0]
            critic_lr = actor_critic_value[1]




            hyperparameters = {
                'env': "Pendulum-v1",
                'std_dev': std_dev,
                'actor_lr': actor_lr,
                'critic_lr': critic_lr,
                'gamma': .99,
                'tau': tau,
                'total_episodes': 1000
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
            i += 1

        
print(trial_params)
