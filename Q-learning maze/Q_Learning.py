""""
most of this code copied from morvan
this part is a Q-learning algorithm for finding a optimal path in a maze
Q(s,a) = Q(s,a) + learning_rate * [reward + gamma* max_a'{Q'(s',a')} - Q(s,a)]
"""
import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
        self.actions = actions      # a list
        self.lr = learning_rate     # learning rate for each episode
        self.gamma = reward_decay   # discount factor
        self.epsilon = e_greedy     # epsilon-greedy police for choosing a action
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)     # initial q_table

    # choose action using epsilon-greedy police depended on the observation
    def choose_action(self, observation):
        # check whether the observation in the state_table
        self.check_state_exist(observation)
        # action select
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some action have same value so random the order
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, reward, s_):
        self.check_state_exist(s_)
        # current state q-table as predict q-table
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = reward # next state is terminal

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update the q table

    def check_state_exist(self, state):
        if state not in self.q_table.index:

            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )