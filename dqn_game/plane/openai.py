# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import time
import math


from gym.envs.registration import register

register(
    id='Plane-v0',
    entry_point='plane:Plane',  # 第一个myenv是文件夹名字，第二个myenv是文件名字，MyEnv是文件内类的名字
)


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make('Plane-v0')
    env.reset()
    env.render()

    while True:
        action = random.randint(0, 4)
        print action
        env.step(action)
        env.render()
        time.sleep(0.02)

if __name__ == '__main__':
    main()