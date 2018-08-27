# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import time
import math

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.02  # starting value of epsilon
FINAL_EPSILON = 0.001  # final value of epsilon
REPLAY_SIZE = 10000  # 经验回放缓存大小
BATCH_SIZE = 200  # 小批量尺寸
TARGET_Q_STEP = 100  # 目标网络同步的训练次数


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = 20
        self.action_dim = 5
        self.hide_layer_inputs = 52
        # 创建Q网络
        self.create_Q_network()
        # 创建训练方法
        self.create_training_method()

        self.target_q_step = TARGET_Q_STEP
        self.create_TargetQ_network()

        # 初始会话
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("models")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_models(self, times):
        self.saver.save(self.session, 'models/snake', global_step=times)

    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, self.hide_layer_inputs])
        b1 = self.bias_variable([self.hide_layer_inputs])
        W2 = self.weight_variable([self.hide_layer_inputs, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, W2) + b2
        # 保存权重
        self.Q_Weihgts = [W1, b1, W2, b2]

    def create_TargetQ_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, self.hide_layer_inputs])
        b1 = self.bias_variable([self.hide_layer_inputs])
        W2 = self.weight_variable([self.hide_layer_inputs, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        # self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # Q Value layer
        self.TargetQ_value = tf.matmul(h_layer, W2) + b2
        self.TargetQ_Weights = [W1, b1, W2, b2]

    def copyWeightsToTarget(self):
        for i in range(len(self.Q_Weihgts)):
            self.session.run(tf.assign(self.TargetQ_Weights[i], self.Q_Weihgts[i]))

    def create_training_method(self):
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)  # mul->matmul
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append([state, one_hot_action, reward, next_state, done])
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def modify_last_reward(self, new_reward):
        v = self.replay_buffer.pop()
        v[2] = new_reward
        self.replay_buffer.append(v)

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # Q_value_batch = self.TargetQ_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

        # 同步目标网络
        if self.time_step % self.target_q_step == 0:
            self.copyWeightsToTarget()

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

        if random.random() <= self.epsilon:
            return np.random.randint(0, 4)
        # return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        return np.argmax(Q_value)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 10000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 1  # The number of experiment test every 100 episode

from gym.envs.registration import register

register(
    id='Plane-v0',
    entry_point='plane:Plane',  # 第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
)



def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make('Plane-v0')
    agent = DQN(env)

    # agent.copyWeightsToTarget()
    state = env.reset()
    state = np.reshape(state, [-1])


    while True:

        env.render()

        action = agent.action(state)
        next_state, reward, done = env.step(action)

        state = np.reshape(next_state, [-1])
        time.sleep(0.1)

        if done:
            break



if __name__ == '__main__':
    main()