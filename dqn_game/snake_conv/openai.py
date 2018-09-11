# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import time
import math

GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.1  # starting value of epsilon
FINAL_EPSILON = 0.001  # final value of epsilon
REPLAY_SIZE = 2000  # 经验回放缓存大小
BATCH_SIZE = 600  # 小批量尺寸


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.action_dim = 4
        # 创建Q网络
        self.create_Q_network()
        # 创建训练方法
        self.create_training_method()

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
        W_conv1 = weight_variable([8, 8, 1, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        W_fc1 = weight_variable([768, 512])
        b_fc1 = bias_variable([512])

        W_fc2 = weight_variable([512, 4])
        b_fc2 = bias_variable([4])

        # input layer
        s = tf.placeholder("float", [None, 16, 12, 1])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 2) + b_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 768])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2
        self.state_input = s


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


    def train_Q_network(self):
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
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


    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]

        self.epsilon = max(self.epsilon - 0.001 / 5000, FINAL_EPSILON)

        if random.random() <= self.epsilon:
            return np.random.randint(0, 4)
        else:
            return np.argmax(Q_value)

    def action(self, state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]
        return np.argmax(Q_value)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------
from gym.envs.registration import register

register(
    id='Snake-v0',
    entry_point='snake:Snake',  # 第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
)


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make('Snake-v0')
    agent = DQN(env)

    step = 0

    while True:

        step += 1
        # initialize task
        state = env.reset()

        # Train
        done = False

        reward_sum = 0
        max_length = 0
        while not done:

            env.render()

            action = agent.egreedy_action(state)
            next_state, reward, done, length = env.step(action)

            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            reward_sum += reward

            max_length = max(max_length, length)

            if done:
                # env.reset()
                break

        print 'step: ', step, "   reward_num: ", reward_sum, "  max_length", max_length, "  epsilon:", agent.epsilon
        if step % 100 == 0:
            agent.save_models(step)


if __name__ == '__main__':
    main()
