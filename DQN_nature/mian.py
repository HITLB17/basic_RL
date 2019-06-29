"""
this part of code is DQN algorithm
most of the code copy from morvan's tutorial: https://morvanzhou.github.io/tutorials
copyright: HITLB
"""
from maze_env import Maze
from DQN import DeepQNetwork
import time


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # choose action
            action = RL.choose_action(observation)

            # after an action , new observation and reward occur
            observation_, reward, done = env.step(action)

            # store transition
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while done = true
            if done:
                break
            step += 1
            time.sleep(0.1)
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()

