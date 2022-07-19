'''
 * File:   Solver.cpp
 * Author: koosha
 *
 * Created on November 26, 2011, 4:55 PM
 *
 * translated to python by Sami beginning June 28, 2022
'''


#because so many globals, best to use OOP
import numpy as np
import time
import random
from pomdp import POMDP


class PairwiseSolver:
    def __init__(self, pomdp, precision = 0.001):
        #initialize numpy arrays as empty, 2d
        self.pomdp = pomdp
        self.precision = precision
        S = len(self.pomdp.states)  # was hardcoded as 5500
        self.pair_values = np.zeros((S, S), float) # pairwise state space, v-values
        self.pair_actions = np.zeros((S, S), int) - 1 #init -1 #pairwise state space, optimal action (policy)
        self.belief = [0] * S
        self.temp = [0] * S

        #initalize numeric variables
        self.nrStates = len(self.pomdp.states) #S
        self.nrActions = len(self.pomdp.actions) #A
        self.nrObservations = len(self.pomdp.observations) #Z
        self.gamma = self.pomdp.discount #init from pomdp file
        self.R_4d = self.pomdp.R #4d #A x S x S x Z
        self.transition = self.pomdp.T #A x S x S
        self.observation = self.pomdp.O #A x S x Z
        self.start = self.pomdp.prior #start_state
        self.Q = np.zeros((len(self.pomdp.actions), len(self.pomdp.states))) #used in MDP
        self.V = np.zeros(len(self.pomdp.states)) #used in MDP
        self.policy = len(self.pomdp.states) #used in MDP

        #make states_actions
        self.states_actions = np.argmax(self.transition, axis=2)

        #make states_observations
        self.states_observations = np.argmax(self.observation, axis=2)

        #make neighbors
        ## TODO: is matrix implementation possible?
        self.neighbors = np.empty((len(self.pomdp.actions), len(self.pomdp.states)), int) #vector in C++ file
        self.neighbors = self.neighbors.tolist()
        for state in range(self.nrStates):
            for action in range(self.nrActions):
                first_found = False
                for end_state in range(self.nrStates):
                    if self.transition[action][state][end_state] > 0:
                        if not first_found:
                            self.neighbors[action][state] = [end_state]
                            first_found = True
                        else:
                            self.neighbors[action][state].append(end_state)


        # generate 3d and 2d Reward
        ## TODO: switch to using R_2d (no matrix implementation possible
        self.R_2d = np.zeros([self.nrActions, self.nrStates])
        self.R_3d = np.zeros([self.nrActions, self.nrStates, self.nrStates])
        for action in range(self.nrActions):
            for start_state in range(self.nrStates):
                for end_state in range(self.nrStates):
                    transition_prob = self.transition[action, start_state, end_state]
                    for obs in range(self.nrObservations):
                        obs_prob = self.pomdp.O[action, end_state, obs]
                        self.R_2d[action, start_state] += obs_prob * \
                                                          transition_prob * \
                                                          self.pomdp.R[action, start_state, end_state, obs]
                        self.R_3d[action, start_state, end_state] += obs_prob * \
                                                                     self.pomdp.R[action, start_state, end_state, obs]
        self.reward = self.R_3d #try to use 2d_reward instead



    def MDP(self):
        '''
        get underlying MDP from given POMDP
        the first n cells are values and the next n are best actions <- this is bad, store separately
        '''

        # implement bellman equation
        self.diff = 1
        while self.diff > self.precision:
            # reformat V for matrix operations
            Vk = np.tile(self.V, self.nrActions * self.nrStates)
            Vk_compute = np.reshape(Vk, (self.nrActions, self.nrStates, self.nrStates))  # AxSxS

            # calculate bellman equation
            subsequent_reward_by_final = np.multiply(self.transition, Vk_compute)  # AxSxS
            subsequent_reward = np.sum(subsequent_reward_by_final, axis=2)  # AxS
            discounted_reward = np.multiply(self.pomdp.discount, subsequent_reward)  # AxS

            # calculate variables
            new_Q = np.add(self.R_2d, discounted_reward)  # AxS
            new_V = np.amax(new_Q, axis=0)  # should be 1 per state #S
            new_policy = np.argmax(new_Q, axis=0)  # should be 1 per state #S
            self.diff = max(np.abs(new_V - self.V))  # int

            # assign variables
            self.Q = new_Q
            self.V = new_V
            self.policy = new_policy



    def SLAP_pairs(self, difference_threshold):
        '''
        implement algorithm 1 (simultaneous localization and planning)
        :param difference_threshold:
        '''
        ## TODO: write as matrix implementation
        one_action_localization = np.zeros((self.nrStates, self.nrStates), float) - 10  # init to MIN value
        indistinguishable_states = []  # fill with tuples of indistinguishable states
        start_time = time.time()

        # pseudocode line 1 - calculate V(S) of the MDP
        self.MDP()

        # pseudocode line 2 - for each pair (s,s') do V(s,s') = R_min
        for s in range(self.nrStates):
            for s_prime in range(s):
                for action in range(self.nrActions):
                    cur_dif = 0

                    # pseudocode line 5 - for each pair (s,s') and (s'', s''') do
                    for i in range(len(self.neighbors[action][s])):  # neighbors stores an index to all possible transitions, transition stores prob
                        for j in range(len(self.neighbors[action][s_prime])):

                            #get s'' and s''' values and observations
                            s_dprime = self.neighbors[action][s][i]
                            o1 = self.states_observations[action][s_dprime]
                            s_tprime = self.neighbors[action][s_prime][j]
                            o2 = self.states_observations[action][s_tprime]

                            #calculate how to check if distinguishable
                            cur_dif += self.transition[action][s][s_dprime] * self.transition[action][s_prime][s_tprime] * \
                                       (self.observation[action][s_dprime][o1] * (1 - self.observation[action][s_tprime][o1]) +
                                        self.observation[action][s_tprime][o2] * (1 - self.observation[action][s_dprime][o2]))

                    #if distinguishable
                    change = False
                    if cur_dif >= 2 * difference_threshold:
                        # equation for lines 11/12 - 0.5[R(s,a) + R(s',a) + discount(V(s) + V(s'))] - this equation seems wrong, V(s'')?
                        temp_value = .5 * (self.reward[action][s][self.states_actions[action][s]] +
                                           self.reward[action][s_prime][self.states_actions[action][s_prime]]) + \
                                           self.gamma * .5 * (self.V[self.states_actions[action][s]] +
                                           self.V[self.states_actions[action][s_prime]])
                        max_value = one_action_localization[s][s_prime]

                        if temp_value > max_value:
                            max_value = temp_value
                            # pseudocode lines 11/12 - assign V(pair_values) and u(pair_Actions)
                            self.pair_values[s][s_prime] = self.pair_values[s_prime][s] = max_value
                            self.pair_actions[s][s_prime] = self.pair_actions[s_prime][s] = action
                            one_action_localization[s][s_prime] = one_action_localization[s_prime][s] = max_value
                            change = True

                # store indistinguishable state pairs
                if not change:
                    indistinguishable_states.append((s, s_prime))

        # pseudocode line 13-17 - repeat until convergence
        for iteration in range(10):

            # pseudocode line 14 - for each indistinguishable pair (s,s') do
            for pair in range(len(indistinguishable_states)):
                s, s_prime = indistinguishable_states[pair]
                max_value = one_action_localization[s][s_prime]
                for ac in range(self.nrActions):
                    # (R((s,s'),a) + discount(sum(V(s'',s_tp)p((s'',s_tp)|(s,s'),a)
                    temp_value = (.5 * (self.reward[ac][s][self.states_actions[ac][s]] +
                                        self.reward[ac][s_prime][self.states_actions[ac][s_prime]]) + self.gamma *
                                        self.pair_values[self.states_actions[ac][s]][self.states_actions[ac][s_prime]])

                    if temp_value > max_value:
                        max_value = temp_value
                        # pseudocode line 15/16 - assign V(pair_values) and u(pair_actions)
                        self.pair_values[s][s_prime] = self.pair_values[s_prime][s] = temp_value
                        self.pair_actions[s][s_prime] = self.pair_actions[s_prime][s] = ac
                        one_action_localization[s][s_prime] = one_action_localization[s_prime][s] = max_value

        #get runtime
        end_time = time.time()
        total_time = end_time - start_time
        print("Total time to run SLAP_pairs was", total_time)

        #format for easy file comparison
        np.savetxt("pair_values.txt", self.pair_values)
        np.savetxt("pair_actions.txt", self.pair_actions)




if __name__ == "__main__":
    pomdp = POMDP("Hallway2.POMDP")
    pairwiseSolver = PairwiseSolver(pomdp)
    pairwiseSolver.SLAP_pairs(0.70)  # initially 0.8

