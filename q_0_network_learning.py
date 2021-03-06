# https://gist.github.com/awjuliani/4d69edad4d0ed9a5884f3cdcf0ea0874#file-q-net-learning-clean-ipynb
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Just disables the warning, doesn't enable AVX/FMA

env = gym.make('FrozenLake-v0')

########################################################
# The Q-Network Approach   # Implementing the network itself
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

########################################################
# Training the network
init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        obs = env.reset()
        rAll = 0
        done = False
        #The Q-Network
        for j in range(99):
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([predict,Qout], feed_dict={inputs1: np.identity(16)[obs:obs + 1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            obs_1, reward, done, _ = env.step(a[0]) #Get new state and reward from environment
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[obs_1:obs_1 + 1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = reward + y * maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W], feed_dict={inputs1: np.identity(16)[obs:obs + 1], nextQ:targetQ})
            rAll += reward
            obs = obs_1
            if done == True: #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break

        jList.append(j)
        rList.append(rAll)

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

plt.plot(rList)
plt.plot(jList)