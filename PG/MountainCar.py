""""
View more on morvan tutorial page: https://morvanzhou.github.io/tutorials/

author: HITLB
If you have any problem, please contact me. 17B904042@stu.hit.edu.cn
"""

import gym
from PG import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False

env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    output_graph=True,
)

reward_r = [0]
for i_episode in range(1000):

    observation = env.reset()

    while True:
        if RENDER: env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)     # reward = -1 in all cases

        #将观测，动作和回报存储起来
        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01

            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))
            reward_r.append(running_reward)
            vt = RL.learn()  # train

            if i_episode == 990:
                RENDER = True
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()

            break
        observation = observation_
plt.plot(reward_r)
plt.show()