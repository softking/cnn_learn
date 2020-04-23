# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import time


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
        # init some parameters
        self.time_step = 0
        # 创建Q网络
        self.create_Q_network()

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

    def create_Q_network(self):
        # network weights
        W_conv1 = weight_variable([8, 8, 1, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])

        W_fc1 = weight_variable([1536, 1024])
        b_fc1 = bias_variable([1024])

        W_fc2 = weight_variable([1024, 4])
        b_fc2 = bias_variable([4])

        # input layer
        s = tf.placeholder("float", [None, 16, 12, 1])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 2) + b_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1536])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2
        self.state_input = s



    def action(self, state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]
        return np.argmax(Q_value)


# ---------------------------------------------------------
from gym.envs.registration import register

register(
    id='Car-v0',
    entry_point='car:Car',  # 第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
)


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make('Car-v0')
    env.set_show()
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

            action = agent.action(state)
            next_state, reward, done, length = env.step(action)

            state = next_state
            reward_sum += reward

            max_length = max(max_length, length)
            time.sleep(0.1)

            if done:
                env.reset()
                break

        print('step: ', step, "   reward_num: ", reward_sum, "  max_length", max_length)


if __name__ == '__main__':
    main()
