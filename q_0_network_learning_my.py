# https://gist.github.com/awjuliani/4d69edad4d0ed9a5884f3cdcf0ea0874#file-q-net-learning-clean-ipynb
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Just disables the warning, doesn't enable AVX/FMA

env = gym.make('FrozenLake-v0')

################################################################################################################
# The Q-Network Approach   # Implementing the network itself
################################################################################################################
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
tf_inp_state = tf.placeholder(shape=[1, 16], dtype=tf.float32)        ##### placeholder
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
tf_out_acts = tf.matmul(tf_inp_state, W)
tf_act_predict = tf.argmax(tf_out_acts, 1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
should_acts = tf.placeholder(shape=[1, 4], dtype=tf.float32)      ##### placeholder
loss = tf.reduce_sum(tf.square(should_acts - tf_out_acts))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

################################################################################################################
# Training the network
################################################################################################################
init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1  #switch for generate random action
num_episodes = 2000
jList = [] #create lists to contain total rewards and steps per episode
rList = []


with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        obs = env.reset() #Reset environment and get first new observation
        rAll = 0
        done = False
        #The Q-Network
        for j in range(99):

            #Choose an action by greedily (with e chance of random action) from the Q-network
            inp = np.identity(16)[obs:obs + 1]
            a,allQ = sess.run([tf_act_predict, tf_out_acts], feed_dict={tf_inp_state: inp})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample() # Random Action
            obs_1, reward, done, _ = env.step(a[0]) #Get new state and reward from environment

            #Obtain the Q' values by feeding the new state through our network
            #Obtain maxQ' and set our target value for chosen action.
            Q1 = sess.run(tf_out_acts, feed_dict={tf_inp_state: np.identity(16)[obs_1:obs_1 + 1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = reward + y * maxQ1

            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel, W], feed_dict={tf_inp_state: np.identity(16)[obs:obs + 1], should_acts:targetQ})
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