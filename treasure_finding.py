"""
copy right HITLB
this code is a simple rl algorithm using Q-learning
the tutorial is https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/
the equation is : Q(s,a) = Q(s,a) + alpha * [reward + gamma* max_a'{Q'(s',a')} - Q(s,a)]
"""
import numpy as np
import pandas as pd
import time

np.random.seed(2) # this is for reproducible, but i don't think so

N_STATES = 6 # the length of the 1 dimension world from the left to the right(goal)
ACTIONS = ['left', 'right'] # the action that the agent can take
EPSILON = 0.9 # epsilon-greedy police for choosing a action
ALPHA = 0.2 # learning rate
GAMMA = 0.9 # discount factor
MAX_EPISODES = 13 # maximum episodes
FRESH_TIME = 0.3 # fresh time for one move


# build the Q-table for q-learning algorithm
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # initial Q-table as full zero
        columns = actions,
    )
    # print table
    print(table)
    # return table
    return table


# choose action using epsilon-greedy police
def choose_action(state, q_table):
    state_action = q_table.iloc[state, :]   #choose q_table at a certain state
    if (np.random.uniform() > EPSILON) or ((state_action == 0).all()):  #act epsilon-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   #　act greedy
        action_name = state_action.idxmax()     # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


# This is how agent will interact with the environment
def get_env_feedback(S, A):
    if A == 'right':    #move right
        if S == N_STATES - 2: # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:       # MOVE LEFT
        R = 0
        if S == 0:
            S_ = S  # REACH THE WALL
        else:
            S_ = S - 1
    return S_, R

# this is how to update the environment
def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']    # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s : total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end = '')
        print('\n')
        time.sleep(1)
        print('\r                                ', end = '')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end ='')
        time.sleep(FRESH_TIME)


# this is RL algorithm loop (Q-learning)
def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1)
            step_counter += 1
        print('\r',q_table)
    return q_table


if  __name__ == "__main__":
    q_table = rl()
    print('\r\n Q-table: \n')
    print(q_table)