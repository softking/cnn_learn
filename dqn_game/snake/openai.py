# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import gym
import time



from gym.envs.registration import register
register(
    id='Snake-v0',
    entry_point='snake:Snake',#第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
)

env = gym.make('Snake-v0')

env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 1000:

    observation, reward, done, _ = env.step(np.random.randint(0, 4))

    reward_sum += reward
    env.render()
    if done:
        random_episodes += 1
        print("reward for this episode was: ", reward_sum)
        reward_sum = 0
        env.reset()