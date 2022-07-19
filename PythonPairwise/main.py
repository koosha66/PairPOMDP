"""
Created on Sun Jan 27 18:48:51 2019
@author: koosha
"""

from PythonPairwise.pomdp import POMDP
import numpy as np
from environment import Environment


class PairwiseHeuristic():
    def __init__(self, pomdp, precision=.001):
        """
        equation: Q_{t+1}^MDP(a, s) = R(a, s) + discount*sum_s'[T(a, s, s')V_t^MDP(s')]
        """
        ## TODO: figure this out

        # init variables
        self.pomdp = pomdp
        self.precision = precision
        self.T = self.pomdp.T
        S = len(self.pomdp.states)
        A = len(self.pomdp.actions)
        Z = len(self.pomdp.observations)
        self.QMDP = np.zeros((A, S))
        self.VMDP = np.zeros(S)
        self.policy = np.zeros(S)
        self.diff = 1

        # generate 2d R
        self.R_2d = np.zeros([A, S])
        for action in range(A):
            for start_state in range(S):
                for end_state in range(S):
                    transition_prob = self.T[action, start_state, end_state]
                    for obs in range(Z):
                        obs_prob = self.pomdp.O[action, end_state, obs]
                        self.R_2d[action, start_state] += obs_prob * \
                                                          transition_prob * \
                                                          self.pomdp.R[action, start_state, end_state, obs]

        # implement bellman equation
        while self.diff > self.precision:
            # reformat V for matrix operations
            Vk = np.tile(self.VMDP, A * S)
            Vk_compute = np.reshape(Vk, (A, S, S))  # AxSxS

            # calculate bellman equation
            subsequent_reward_by_final = np.multiply(self.T, Vk_compute)  # AxSxS
            subsequent_reward = np.sum(subsequent_reward_by_final, axis=2)  # AxS
            discounted_reward = np.multiply(self.pomdp.discount, subsequent_reward)  # AxS

            # calculate variables
            new_QMDP = np.add(self.R_2d, discounted_reward)  # AxS
            new_VMDP = np.amax(new_QMDP, axis=0)  # should be 1 per state #S
            new_policy = np.argmax(new_QMDP, axis=0)  # should be 1 per state #S
            self.diff = max(np.abs(new_VMDP - self.VMDP))  # int

            # assign variables
            self.QMDP = new_QMDP
            self.VMDP = new_VMDP
            self.policy = new_policy


    def solve(self):
        """
        solve and calculate the total reward
        for one run
        """
        total_reward = 0
        environment = Environment(self.pomdp)
        time_step = 0
        Max_abs_reward = np.max(np.abs(self.pomdp.R))
        cur_belief = np.array(self.pomdp.prior).reshape(1, len(self.pomdp.prior))
        while (Max_abs_reward * (self.pomdp.discount ** time_step) > self.precision):
            # each iteration
            action = self.chooseAction(cur_belief)
            reward, obs = environment.act(action)
            total_reward += reward * (self.pomdp.discount ** time_step)
            cur_belief = self.updateBelief(cur_belief, action, obs)
            time_step += 1
        return total_reward

    def updateBelief(self, current_belief, action, observation):
        """
        update the belief
        output: the updated belief based on the current belief, action, and observation
        equation: b_t(s) = 1/P * O(a, s', z) * sum_s[T(a, s, s') * b_{t-1}(s)]
        """
        ## TODO: figure this out

        return current_belief / np.sum(current_belief)  # if self.pomdp.O[] was0, now we have nan, BAD

    def chooseAction(self, cur_belief):
        """
        return the best action according to the current belief
        """
        ## TODO: figure this out
        return best_action


if __name__ == "__main__":
    num_simulations = 5000
    filename_env = 'Hallway.POMDP'  ### or Hallway2.pomdp; Figure 2 and 3 of https://people.cs.umass.edu/~barto/courses/cs687/Cassandra-etal-POMDP.pdf
    # or AEMS2/examples/envs/hallway.pomdp, must change terminal domain
    pomdp = POMDP(filename_env)
    pomdp.print_summary()
    pairwise_solver = PairwiseHeuristic(pomdp)
    rewards = np.zeros(num_simulations)
    for sim in range(num_simulations):
        rewards[sim] = pairwise_solver.solve()

    print("Average total discounted reward of all", num_simulations, "simulations (Hallway):", np.mean(rewards))

