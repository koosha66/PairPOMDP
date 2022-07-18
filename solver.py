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
        self.states_actions = np.empty((self.nrActions, self.nrStates), int) #argmax of belief_state, s* from paper
        for state in range(self.nrStates):
            for action in range(self.nrActions):
                state_distribution = self.transition[action][state]
                self.states_actions[action][state] = np.argmax(state_distribution)

        #make states_observations
        self.states_observations = np.zeros((self.nrActions, self.nrStates), int)
        for state in range(self.nrStates):
            for action in range(self.nrActions):
                observation_distribution = self.observation[action][state]
                self.states_observations[action][state] = np.argmax(observation_distribution)

        #make neighbors
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
        self.reward = self.R_3d



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

        print(self.policy)



    def SLAP_pairs(self, difference_threshold):
        '''
        implement algorithm 1 (simultaneous localization and planning)
        :param difference_threshold:
        '''
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
                    # (R((s,s'),a) + discount(sum(V(s'',s''')p((s'',s''')|(s,s'),a)
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


    def online_planner(self, compare_ratio):
        '''
        implement Algorithm 2, solver, online planning method
        :param compare_ratio:
        '''
        ## TODO: review, clean, fix (reward is still 0)
        ## TODO: format as matrices
        iterations = 151 #NUM_ITERATIONS in C++
        max_utility = -10 #MIN in C++
        valid_action = [False] * self.nrActions
        actions = [-1] * iterations

        self.belief = np.multiply(self.start, 1)
        real_state = self.choose_start_state()
        visited_states = [real_state]

        start_time = time.time()

        for iteration in range(iterations):
            list_states = []
            real_states = []

            # pseudocode line 1 - maxBel = max_s b_k(s)
            real_states.append(real_state)
            max_bel = np.argmax(self.belief, axis=0)

            # pseudocode line 2 - S' = {s|b(s) > maxBel/comp_ratio
            for state in range(self.nrStates):
                if self.belief[state] > self.belief[max_bel] / compare_ratio:
                    list_states.append(state)  # list_states = set S'

            # pseudocode line 4/5 - if |S'| == 1 then a is optimal action
            if len(list_states) == 1:
                action = self.policy[list_states[0]]

            # if multiple likely states (loop beginning pseudocode line 6)
            else:
                # pseudocode line 3 - A' = {a|a = u(s,s'), s, s' in S'}
                for i in range(len(list_states)):
                    for j in range(i):
                        s = list_states[i]
                        s_prime = list_states[j]
                        if self.pair_actions[s][s_prime] >= 0:
                            valid_action[self.pair_actions[s][s_prime]] = True

                # pseudocode line 7 - for each a in A' do
                for ac in range(self.nrActions):
                    if valid_action[ac]:
                        utility = 0

                        # pseudocode line 8 - for each s in S' do s* = argmax_s' p(s'|s, a)
                        for i in range(len(list_states)):
                            for j in range(i):
                                s1 = list_states[i]
                                s2 = list_states[j]
                                # pseudocode line 8.1 - H(a) = sum[(0.5 * (R(s,a) + R(s',a)) + gamma V(s*,s'*))b(s)b(s')]
                                utility += self.belief[s1] * self.belief[s2] * \
                                           (.5 * self.reward[ac][s1][self.states_actions[ac][s1]] + .5 *
                                            self.reward[ac][s2][self.states_actions[ac][s2]] + self.gamma *
                                            self.pair_values[self.states_actions[ac][s1]][self.states_actions[ac][s2]])

                        # pseudocode line 9 - a = argmax_a H(a)
                        if utility > max_utility:
                            max_utility = utility
                            action = ac

            #act, observe, and update belief
            obs, real_state = self.act_and_observe(real_state, action)
            actions[iteration] = action
            visited_states.append(real_state)
            self.belief = self.update_bel(action, obs)

        #after iterations complete. get reward and time
        total_reward = 0
        discount = 1
        for iteration in range(iterations - 1):
            discount = discount * self.gamma
            s = visited_states[iteration]
            s_prime = visited_states[iteration+1]
            total_reward += discount * self.reward[actions[iteration]][s][s_prime]
        end_time = time.time()

        return total_reward


    def act_and_observe(self, real_state, action):
        '''
        make the action, observe new state
        :param real_state:
        :param action:
        '''
        '''
        rand_max = 32762
        r = random.randrange(0, rand_max) % 32000
        r = r / 32000
        sum = 0
        for i in range(self.nrStates-1):
            sum = sum + self.transition[action][i][real_state]
            if sum > r:
                break
        real_state = i
        sum = 0
        r = random.randrange(0, rand_max) % 32000
        r = r / 32000
        for i in range(self.nrObservations):
            sum = sum + self.observation[action][real_state][i]
            if sum > r:
                return i, real_state
        return i, real_state
        '''

        # from QMDP.environment
        # NOTE: making this change removed /0 error but decreased reward

        # determine start state
        start_state = real_state

        # determine end state and update current
        transition_prob = self.pomdp.T[action, start_state]
        end_state = np.random.choice(np.arange(0, len(transition_prob)), p=transition_prob)

        # find observation distribution
        observation_distribution = self.pomdp.O[action, end_state]
        observation = np.random.choice(np.arange(0, len(observation_distribution)), p=observation_distribution)

        return observation, end_state



    def update_bel(self, action, o):
        '''
        from QMDP.py implementation, confirmed works
        :return: new belief state
        '''
        #compute transitions
        current_belief = np.matmul(self.belief, self.pomdp.T[action])

        #multiply by observation probability
        current_belief = current_belief * self.pomdp.O[action, :, o]

        return current_belief / np.sum(current_belief) #if self.pomdp.O[] was 0, now we have nan, BAD


    def choose_start_state(self):
        #confirmed OK from previous QMDP code
        probability_distribution = self.start
        start_state = np.random.choice(np.arange(0, self.nrStates), p=probability_distribution)
        return start_state




if __name__ == "__main__":
    pomdp = POMDP("Hallway2.POMDP")
    pairwiseSolver = PairwiseSolver(pomdp)
    pairwiseSolver.SLAP_pairs(0.70) #initially 0.8

    num_sims = 100 #for testing
    #for j in range(10): #what is this 10
    #    sum_reward = 0
    #    #below was nested
    for i in range(num_sims):
        sum_reward = 0
        temp_reward = pairwiseSolver.online_planner(8) #initally 19
        sum_reward += temp_reward

    print("Average total reward after", num_sims, "simulations", sum_reward/num_sims)



