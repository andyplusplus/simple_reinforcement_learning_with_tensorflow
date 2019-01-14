# https://gist.github.com/awjuliani/9024166ca08c489a60994e529484f7fe#file-q-table-learning-clean-ipynb

import gym
import numpy as np
import my_utility

env = gym.make('FrozenLake-v0')

## row --> state
## col --> action
Q_table = np.zeros([env.observation_space.n, env.action_space.n]) #(16, 4) Initialize table with all zeros
# Set learning parameters
lr = .8 # learn rate
y = .95 # discounted future reward
num_episodes_2000 = 2000
rList = [] #create lists to contain total rewards and steps per episode
for i in range(num_episodes_2000):
    obs = env.reset() #Reset environment and get first new observation
    rAll = 0
    done = False
    #The Q-Table learning algorithm
    for j in range(99): #Choose an action by greedily (with noise) picking from Q table
        args_q_table = Q_table[obs, :] + \
                       np.random.randn(1, env.action_space.n) * (1. / (i + 1))
        a = np.argmax(args_q_table)

        obs_1, reward, done, _ = env.step(a) #Get new state and reward from environment

        # Update Q-Table with new knowledge:
        # q(s,a) = r + y(max(q(s', a'))
        discounted_future_reward = y * np.max(Q_table[obs_1, :])
        new_reward = reward + discounted_future_reward
        Q_table[obs, a] = Q_table[obs, a] + lr * (new_reward - Q_table[obs, a])

        #Q_table[obs, a] = Q_table[obs, a] \
        #                  + lr * (reward + y * np.max(Q_table[obs_1, :])
        #                          - Q_table[obs, a])

        rAll += reward
        obs = obs_1
        if done == True:
            break
    rList.append(rAll)

print("Score over time: " + str(sum(rList) / num_episodes_2000))

print("Final Q-Table Values")
print(Q_table)
print("\n reward list")
print(rList)
my_utility.list_printer(rList)