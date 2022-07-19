"""
Simulate the enviroment for a POMDP model

@author: koosha
"""
import numpy as np


class Environment:
    def __init__(self, pomdp):
        """
        choose the initial state as the current state
        """
        self.pomdp = pomdp

        ## TODO: figure this out
        self.current_state = 0

    def act(self, action):
        """
        Perform the given action
        update self.current_state
        return reward and observation
        """
        ## TODO: figure this out

        # determine start state
        start_state = self.current_state  # environment chooses most likely state

        # determine end state and update current
        transition_prob = self.pomdp.T[action, start_state]
        end_state = np.random.choice(np.arange(0, len(transition_prob)), p=transition_prob)
        self.current_state = end_state

        # find observation distribution
        observation_distribution = self.pomdp.O[action, end_state]
        observation = np.random.choice(np.arange(0, len(observation_distribution)), p=observation_distribution)

        # determine reward
        reward = self.pomdp.R[action, start_state, end_state, observation]
        if reward == 0 and np.where(self.pomdp.T[:, end_state, end_state] < 1)[0].size == 0:
            reward = None

        return reward, observation


