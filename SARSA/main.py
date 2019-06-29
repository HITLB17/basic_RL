"""
this is the main code in this project
main of the code copy from https://morvanzhou.github.io/tutorials/
author: HITLB
If you have any problem, please contact me. 17B904042@stu.hit.edu.cn
"""

from maze_env import Maze
from SARSA import SarsaTable


def update():
    for episode in range(20):
        # initial observation
        observation = env.reset()
        # choose an action based on observation e-greedy
        action = RL.choose_action(str(observation))

        while True:

            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition(s,a,r,s_,a_)
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action for next step
            observation = observation_
            action = action_

            if done:
                break
        print(RL.q_table)
    print('q-table is\n')
    print(RL.q_table)
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(20, update)
    env.mainloop()